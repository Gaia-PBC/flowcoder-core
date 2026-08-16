"""Flowcoder engine — transparent Claude CLI proxy with flowchart execution.

Behaves identically to `claude -p --input-format stream-json --output-format
stream-json`.  Proxies all stdin/stdout transparently to an inner Claude
subprocess.  When a user message contains a slash command that matches a
known flowchart command, the engine takes over, runs the flowchart, emits
structured events, then resumes proxying.

Architecture
------------
A background ``_MessageRouter`` continuously reads from engine stdin and
dispatches messages into two asyncio queues:

* **control_response_queue** — forwarded to inner Claude immediately by a
  background drainer task, or consumed by ``_handle_control_request``
  during flowchart takeover.
* **message_queue** — everything else (user messages, control_requests,
  shutdown, etc.) consumed by the main loop.

This eliminates deadlocks: control_responses always reach inner Claude
regardless of what the main loop is doing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time

import flowcoder_flowchart as fc_lib
from opentelemetry import context as otel_context
from opentelemetry import trace

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.propagate import extract
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL_SDK = True
except ImportError:
    _HAS_OTEL_SDK = False

from .cli import build_inner_claude_cmd, build_inner_env, build_variables, parse_args
from .protocol import ProtocolHandler
from .resolver import CommandNotFoundError, resolve_command
from .session import BaseSession, ClaudeSession
from .session_factory import SessionFactory
from .subprocess import ClaudeProcess, find_claude
from .walker import ExecutionError, GraphWalker

_log = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


def _init_tracing() -> Any:
    """Set up OTel TracerProvider with OTLP/gRPC exporter if SDK is available."""
    if not _HAS_OTEL_SDK:
        _log.debug("OpenTelemetry SDK not installed, tracing disabled")
        return None
    endpoint = os.environ.get("OTEL_ENDPOINT", "http://localhost:4317")
    resource = Resource.create({"service.name": "flowcoder-engine"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    _log.info("OpenTelemetry tracing initialized (endpoint=%s)", endpoint)
    return provider


def _extract_trace_context(msg: dict) -> otel_context.Context | None:
    """Extract and remove _trace_context from a message, returning an OTel context."""
    if not _HAS_OTEL_SDK:
        msg.pop("_trace_context", None)
        return None
    carrier = msg.pop("_trace_context", None)
    if carrier and isinstance(carrier, dict):
        return extract(carrier)
    return None


_TIMESTAMP_RE = __import__("re").compile(
    r"^\[[\d\-T:+ ]+(?:UTC)?\]\s*"
)


def _parse_slash_command(text: str) -> tuple[str, str] | None:
    """Parse a slash command from text like '/story "a dragon"'.

    Handles optional timestamp prefix from the bot framework, e.g.
    '[2026-03-02 20:12:50 UTC] /story "a dragon"'.

    Returns (command_name, args_string) or None if not a slash command.
    """
    text = text.strip()
    # Strip optional timestamp prefix (e.g. "[2026-03-02 20:12:50 UTC] ")
    text = _TIMESTAMP_RE.sub("", text)
    if not text.startswith("/"):
        return None
    parts = text[1:].split(None, 1)
    if not parts:
        return None
    return (parts[0], parts[1] if len(parts) > 1 else "")


class _StdinReader:
    """Async line reader for stdin."""

    def __init__(self) -> None:
        self._reader: asyncio.StreamReader | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._reader = asyncio.StreamReader()

        async def _pump() -> None:
            try:
                while True:
                    line = await loop.run_in_executor(
                        None, sys.stdin.buffer.readline
                    )
                    if not line:
                        self._reader.feed_eof()
                        break
                    self._reader.feed_data(line)
            except (OSError, ValueError):
                self._reader.feed_eof()

        asyncio.create_task(_pump())

    async def read_message(self) -> dict | None:
        """Read one JSON message from stdin.  Returns None on EOF."""
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                return None
            text = line.decode().strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue


class _MessageRouter:
    """Reads stdin and routes messages into typed queues.

    Runs a background task that continuously reads from ``_StdinReader``
    and dispatches each message to either the control_response queue or
    the general message queue.  This ensures control_responses are never
    blocked behind unread user messages (or vice-versa).
    """

    def __init__(self, stdin_reader: _StdinReader) -> None:
        self._stdin = stdin_reader
        self.message_queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self.control_response_queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._route())

    async def _route(self) -> None:
        """Read stdin forever, dispatching to queues."""
        while True:
            msg = await self._stdin.read_message()
            if msg is None:
                # Signal EOF to both queues
                await self.message_queue.put(None)
                await self.control_response_queue.put(None)
                return
            if msg.get("type") == "control_response":
                await self.control_response_queue.put(msg)
            else:
                await self.message_queue.put(msg)

    async def read_message(self) -> dict | None:
        """Read the next non-control-response message."""
        return await self.message_queue.get()

    async def read_control_response(self) -> dict | None:
        """Read the next control_response."""
        return await self.control_response_queue.get()


async def main() -> None:
    global _tracer
    args = parse_args()
    protocol = ProtocolHandler()

    otel_provider = _init_tracing()
    _tracer = trace.get_tracer(__name__)

    # Find claude binary
    try:
        claude_path = args.claude_path or find_claude()
    except FileNotFoundError as e:
        protocol.emit_result(str(e), is_error=True)
        sys.exit(1)

    # Build inner Claude command and environment from parsed args.
    claude_cmd = build_inner_claude_cmd(args, claude_path)
    inner_env = build_inner_env(args)
    inner_cwd = args.cwd or ""

    # Start the inner Claude process
    process = ClaudeProcess()
    await process.start(claude_cmd, inner_env, inner_cwd)
    protocol.log("Inner claude process started")

    # Signal handling
    shutdown_event = asyncio.Event()

    def handle_signal(signum: int, frame: object) -> None:
        protocol.log(f"Received signal {signum}, shutting down...")
        shutdown_event.set()
        if process.is_running and process._proc is not None:
            try:
                process._proc.kill()
            except (ProcessLookupError, OSError):
                pass

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Set up stdin reader and message router
    stdin_reader = _StdinReader()
    await stdin_reader.start()
    router = _MessageRouter(stdin_reader)
    await router.start()

    # Build session for flowchart execution.
    session: BaseSession = ClaudeSession(
        "main",
        claude_cmd,
        protocol=protocol,
        control_callback=lambda req: _handle_control_request(
            req, protocol, router
        ),
    )
    # Attach the already-running process to the session
    session._process = process  # type: ignore[attr-defined]

    # Build session factory for spawned agents.
    # Both "claude" and "codex" backends create ClaudeSession; codex routing
    # is handled by ANTHROPIC_BASE_URL/ANTHROPIC_MODEL env vars inherited
    # from the engine's environment.
    def _make_claude_session(name: str, model: str | None) -> ClaudeSession:
        cmd = [*claude_cmd, "--model", model] if model else list(claude_cmd)
        return ClaudeSession(name=name, claude_cmd=cmd, protocol=protocol)

    factory = SessionFactory()
    factory.register("claude", _make_claude_session)
    factory.register("codex", _make_claude_session)

    try:
        while not shutdown_event.is_set():
            msg = await router.read_message()
            if msg is None:
                break

            msg_type = msg.get("type", "")

            if msg_type == "shutdown":
                protocol.log("Shutdown requested")
                break

            if msg_type == "user":
                # Extract propagated trace context (strips _trace_context from msg)
                parent_ctx = _extract_trace_context(msg)
                ctx_token = otel_context.attach(parent_ctx) if parent_ctx else None

                try:
                    content = msg.get("message", {}).get("content", "")
                    if not content:
                        continue

                    parsed = _parse_slash_command(content)
                    if parsed:
                        cmd_name, cmd_args = parsed
                        # Try to resolve as a flowchart command
                        try:
                            cmd = resolve_command(
                                cmd_name, args.search_paths
                            )
                        except CommandNotFoundError:
                            # Forward to claude as-is
                            protocol.log(
                                f"Unknown command /{cmd_name}, proxying to claude"
                            )
                            await _proxy_turn(process, protocol, msg, router)
                            continue

                        # Known command — takeover for flowchart execution.
                        # Drain stale control_responses left from prior proxy turns.
                        _drained = 0
                        while not router.control_response_queue.empty():
                            router.control_response_queue.get_nowait()
                            _drained += 1
                        if _drained:
                            protocol.log(
                                f"Drained {_drained} stale control_response(s)"
                            )

                        protocol.log(f"Flowchart takeover: /{cmd_name} {cmd_args}")
                        await _run_flowchart_takeover(
                            session, cmd, cmd_name, cmd_args, protocol, args, factory,
                            router=router,
                        )
                    else:
                        # Normal message — proxy to inner claude
                        await _proxy_turn(process, protocol, msg, router)
                finally:
                    if ctx_token is not None:
                        otel_context.detach(ctx_token)

            elif msg_type == "control_request":
                msg.pop("_trace_context", None)  # strip before forwarding
                subtype = msg.get("request", {}).get("subtype", "")
                if subtype == "initialize":
                    request_id = msg.get("request_id", "")
                    protocol.emit({
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": request_id,
                            "response": {},
                        },
                    })
                else:
                    await process.write(msg)
                    await _forward_until_control_response(
                        process, protocol
                    )

            else:
                # Forward any other message types to inner claude
                msg.pop("_trace_context", None)
                await process.write(msg)

    finally:
        await session.stop()
        await process.stop()
        if otel_provider:
            otel_provider.shutdown()
        protocol.log("Shutdown complete")



async def _proxy_turn(
    process: ClaudeProcess,
    protocol: ProtocolHandler,
    user_msg: dict,
    router: _MessageRouter,
) -> None:
    """Forward a user message to inner claude and proxy all output until result.

    Spawns a background drainer that forwards control_responses from the
    router to inner Claude, preventing deadlocks during MCP init and tool
    permission requests.
    """
    with _tracer.start_as_current_span("proxy_turn"):
        await process.write(user_msg)

        drainer = asyncio.create_task(
            _drain_control_responses(process, router)
        )
        try:
            while True:
                data = await process.read()
                if data is None:
                    return
                protocol.emit(data)
                if data.get("type") == "result":
                    return
        finally:
            drainer.cancel()
            try:
                await drainer
            except asyncio.CancelledError:
                pass


async def _drain_control_responses(
    process: ClaudeProcess,
    router: _MessageRouter,
) -> None:
    """Forward control_responses from the router to inner Claude until cancelled.

    Runs as a background task during proxy turns.  The router has already
    classified the message, so we just read and forward.
    """
    while True:
        msg = await router.read_control_response()
        if msg is None:
            return
        await process.write(msg)


async def _forward_until_control_response(
    process: ClaudeProcess,
    protocol: ProtocolHandler,
) -> None:
    """Forward inner Claude output until a control_response appears.

    Used when the outer client sends a control_request (e.g. set_model)
    to inner Claude.  Control_responses from stdin are handled by the
    router's background drainer, so we only need to read stdout here.
    """
    while True:
        data = await process.read()
        if data is None:
            return
        protocol.emit(data)
        if data.get("type") == "control_response":
            return


async def _pump_router_to_inbox(
    router: _MessageRouter, protocol: ProtocolHandler
) -> None:
    """Feed the walker's inbox during a flowchart takeover.

    In engine mode stdin is drained by ``_MessageRouter`` into its
    ``message_queue``, but the walker reads user input from the
    ``ProtocolHandler`` inbox, and the main loop that normally bridges the two
    is blocked inside ``walker.run()`` for the whole takeover.  This keeps that
    bridge alive so input blocks receive their responses.
    """
    while True:
        msg = await router.read_message()
        if msg is None:
            # EOF/shutdown sentinel: never push None into the inbox (the walker
            # calls .get on the message).  Hand it back so the main loop sees
            # the shutdown once takeover unwinds, then stop pumping.
            router.message_queue.put_nowait(None)
            return
        protocol.push_message(msg)


async def _run_flowchart_takeover(
    session: BaseSession,
    cmd: fc_lib.Command,
    cmd_name: str,
    cmd_args: str,
    protocol: ProtocolHandler,
    args: object,
    session_factory: SessionFactory | None = None,
    router: _MessageRouter | None = None,
) -> None:
    """Execute a flowchart command in takeover mode."""
    block_count = len(cmd.flowchart.blocks)
    with _tracer.start_as_current_span(
        "flowchart.takeover",
        attributes={"command.name": cmd_name, "command.args": cmd_args, "flowchart.block_count": block_count},
    ) as span:
        start_time = time.monotonic()

        # Validate flowchart
        validation = fc_lib.validate(cmd.flowchart)
        if not validation.valid:
            protocol.emit_result(
                f"Flowchart validation failed: {validation.errors}",
                is_error=True,
            )
            span.set_status(trace.StatusCode.ERROR, "validation failed")
            return

        # Build variables
        try:
            variables = build_variables(cmd_args, cmd.arguments)
        except ValueError as e:
            protocol.emit_result(str(e), is_error=True)
            span.set_status(trace.StatusCode.ERROR, str(e))
            return

        protocol.emit_flowchart_start(
            cmd_name, cmd_args, block_count, session=session.name
        )

        walker = GraphWalker(
            cmd.flowchart,
            session,
            variables,
            protocol,
            max_blocks=args.max_blocks,
            search_paths=args.search_paths or [],
            session_factory=session_factory,
        )

        cost_before = session.total_cost

        # During takeover the main loop is parked on this call and cannot drain
        # the router's message_queue, so bridge it into the walker's inbox here —
        # otherwise input blocks never receive their responses.
        pump: asyncio.Task[None] | None = None
        if router is not None:
            pump = asyncio.create_task(_pump_router_to_inbox(router, protocol))

        try:
            result = await walker.run()
            duration_ms = int((time.monotonic() - start_time) * 1000)
            flowchart_cost = session.total_cost - cost_before

            protocol.emit_flowchart_complete(
                status=result.status,
                duration_ms=duration_ms,
                cost_usd=flowchart_cost,
                blocks_executed=len(result.log),
                session_id=session.session_id or "",
                session=session.name,
            )

            protocol.emit_result(
                json.dumps(result.variables),
                is_error=result.status != "completed",
                duration_ms=duration_ms,
                num_turns=len(result.log),
                total_cost_usd=flowchart_cost,
                session_id=session.session_id or "flowchart",
            )
            span.set_attributes({"flowchart.status": result.status, "flowchart.duration_ms": duration_ms})

        except ExecutionError as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            protocol.emit_flowchart_complete(
                status="error", duration_ms=duration_ms,
                session_id=session.session_id or "",
                session=session.name,
            )
            protocol.emit_result(
                str(e), is_error=True,
                session_id=session.session_id or "flowchart",
            )
            span.set_status(trace.StatusCode.ERROR, str(e))

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            protocol.emit_flowchart_complete(
                status="error", duration_ms=duration_ms,
                session_id=session.session_id or "",
                session=session.name,
            )
            protocol.emit_result(
                f"Unexpected error: {e}", is_error=True,
                session_id=session.session_id or "flowchart",
            )
            span.set_status(trace.StatusCode.ERROR, str(e))
        finally:
            if pump is not None:
                pump.cancel()
                try:
                    await pump
                except asyncio.CancelledError:
                    pass
                # Hand back any messages routed into the inbox but not consumed
                # by the walker (e.g. arrived during a non-input block) so the
                # main loop still processes them after takeover.
                assert router is not None
                for leftover in protocol.drain_inbox():
                    router.message_queue.put_nowait(leftover)


async def _handle_control_request(
    request: dict,
    protocol: ProtocolHandler,
    router: _MessageRouter,
) -> dict:
    """Relay a control request from inner claude to the client.

    Emits the request on stdout, waits for a control_response from the
    router's control_response queue.
    """
    protocol.emit(request)

    # Wait for the matching control_response from the client.
    # Discard stale responses whose request_id doesn't match — they
    # are left over from a previous interaction and would cause inner
    # Claude to receive a mismatched response, silently deadlocking it.
    request_id = request.get("request_id", "")
    while True:
        response = await router.read_control_response()
        if response is None:
            # Client disconnected — deny
            return {
                "type": "control_response",
                "response": {
                    "request_id": request_id,
                    "allowed": False,
                },
            }
        resp_id = response.get("response", {}).get("request_id", "")
        if resp_id == request_id or not request_id:
            return response
        # Mismatched — discard and keep waiting
        protocol.log(
            f"Discarded stale control_response "
            f"(got {resp_id!r}, expected {request_id!r})"
        )


def main_sync() -> None:
    """Synchronous entry point for the console script."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
