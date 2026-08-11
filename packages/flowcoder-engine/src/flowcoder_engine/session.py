"""Session — manages AI subprocess lifecycles.

BaseSession defines the backend-agnostic interface.
ClaudeSession implements it for the Claude CLI (stream-json protocol).

Uses the inline ClaudeProcess for subprocess management.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

from opentelemetry import trace

from .subprocess import (
    QUERY_READ_TIMEOUT,
    ClaudeProcess,
    ReadInterruptedError,
    ReadTimeoutError,
)

_tracer = trace.get_tracer(__name__)

if TYPE_CHECKING:
    from .protocol import ProtocolHandler

# Async callback that the proxy core provides for relaying control requests.
# Takes a control_request dict, returns the control_response dict.
ControlCallback = Callable[
    [dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]
]


@dataclass(frozen=True)
class TokenUsage:
    """Token counts broken out by kind, mirroring the Claude CLI's
    ``result.usage`` block.

    On a session these are *cumulative* totals.  Unlike ``total_cost_usd``
    (which the CLI reports cumulatively), the CLI's ``usage`` block is
    reported *per turn*, so ClaudeSession accumulates it by SUMMING each
    query's usage rather than taking a delta (see ``_accumulate_usage``).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class QueryResult:
    """Result from a session query."""

    __slots__ = (
        "cost_usd",
        "duration_ms",
        "response_text",
        "structured_output",
        "token_usage",
    )

    def __init__(
        self,
        response_text: str = "",
        structured_output: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        token_usage: TokenUsage | None = None,
    ) -> None:
        self.response_text = response_text
        self.structured_output = structured_output
        self.cost_usd = cost_usd
        self.duration_ms = duration_ms
        # Per-turn usage for THIS query; the session sums these into its
        # cumulative ``token_usage``.
        self.token_usage = token_usage if token_usage is not None else TokenUsage()


def _clean_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of os.environ for the inner Claude CLI.

    Sets the env vars that make inner Claude use the SDK control protocol
    (for tool permissions and MCP), without depending on the full
    claude-code-sdk package.

    Optional `overrides` are applied last (e.g. ANTHROPIC_BASE_URL for
    routing through anthropic-proxy-rs).
    """
    from .cli import SDK_VERSION

    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"
    env["CLAUDE_AGENT_SDK_VERSION"] = SDK_VERSION
    if overrides:
        env.update(overrides)
    return env


class BaseSession(ABC):
    """Backend-agnostic session interface.

    All session implementations (e.g. ClaudeSession) must implement
    these methods.  Walker and other engine components depend only on
    this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable session name (e.g. 'main', 'spawned-lint')."""
        ...

    @property
    @abstractmethod
    def session_id(self) -> str | None:
        """Backend-assigned session ID, or None if not yet started."""
        ...

    @property
    @abstractmethod
    def total_cost(self) -> float:
        """Cumulative cost in USD across all queries in this session."""
        ...

    @property
    def token_usage(self) -> TokenUsage:
        """Cumulative token counts across all queries in this session.

        Concrete (not abstract) with an all-zero default so backends that do
        not report token usage need no code; ClaudeSession overrides it with
        real accounting.  Summed per-turn, unlike ``total_cost``.
        """
        return TokenUsage()

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the underlying subprocess is alive."""
        ...

    @abstractmethod
    def clone(self, name: str, cwd: str | None = None) -> BaseSession:
        """Create a new session with the same config but a different name.

        ``cwd`` overrides the working directory when provided; None inherits
        the parent's (spawn uses this to isolate a child on disk).
        """
        ...

    @abstractmethod
    async def start(self) -> None:
        """Spawn the backend subprocess."""
        ...

    @abstractmethod
    async def query(
        self,
        prompt: str,
        block_id: str = "",
        block_name: str = "",
    ) -> QueryResult:
        """Send a prompt and return the complete response."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear conversation history (typically by restarting)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Terminate the subprocess."""
        ...

    @abstractmethod
    def with_model(self, model: str) -> BaseSession:
        """Return a new session configured to use a different model."""
        ...

    async def set_permission_mode(self, mode: str) -> None:
        """Change the permission mode on a running session.

        No-op by default — only backends that support runtime permission
        changes (e.g. Claude CLI) need to override this.
        """


def _strip_continuity_flags(cmd: list[str]) -> list[str]:
    """Return ``cmd`` with conversation-continuity flags removed.

    Drops ``--resume [id]``, ``--continue``, and ``--session-id [id]``.  A
    refresh (``clear()``) means "start a new conversation", so these flags —
    which a host such as axi injects at launch for wake-continuity — must not
    survive into the respawned process, or the "cleared" session would simply
    reload the prior conversation.
    """
    out: list[str] = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok in ("--resume", "--session-id"):
            # Drop the flag and its value, unless the next token is another flag
            # (i.e. the flag was used without an explicit value).
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if tok == "--continue":
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


class ClaudeSession(BaseSession):
    """A single Claude CLI subprocess speaking stream-json protocol."""

    def __init__(
        self,
        name: str,
        claude_cmd: list[str],
        protocol: ProtocolHandler | None = None,
        control_callback: ControlCallback | None = None,
        env_overrides: dict[str, str] | None = None,
        cwd: str | None = None,
        read_timeout: float = QUERY_READ_TIMEOUT,
    ) -> None:
        self._name = name
        self._session_id: str | None = None
        self._total_cost: float = 0.0
        self._last_cli_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cache_creation_tokens: int = 0
        self._total_cache_read_tokens: int = 0
        self._claude_cmd = claude_cmd
        self._protocol = protocol
        self._control_callback = control_callback
        self._env_overrides = dict(env_overrides) if env_overrides else None
        self._cwd = cwd
        self._read_timeout = read_timeout
        self._process: ClaudeProcess | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    # -- BaseSession properties --

    @property
    def name(self) -> str:
        return self._name

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def token_usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self._total_input_tokens,
            output_tokens=self._total_output_tokens,
            cache_creation_tokens=self._total_cache_creation_tokens,
            cache_read_tokens=self._total_cache_read_tokens,
        )

    def _accumulate_usage(self, usage: dict[str, Any]) -> TokenUsage:
        """Add one query result's per-turn token usage to the running totals.

        The CLI's ``result.usage`` reports *this turn's* tokens (verified
        against real CLI output: the block is per-turn, not cumulative like
        ``total_cost_usd``), so we SUM across queries rather than diffing.
        Returns the per-turn usage so the caller can stamp its QueryResult.
        """
        turn = TokenUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_creation_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        )
        self._total_input_tokens += turn.input_tokens
        self._total_output_tokens += turn.output_tokens
        self._total_cache_creation_tokens += turn.cache_creation_tokens
        self._total_cache_read_tokens += turn.cache_read_tokens
        return turn

    def clone(self, name: str, cwd: str | None = None) -> ClaudeSession:
        """Create a new ClaudeSession with the same config but a different name.

        ``cwd`` overrides the working directory when provided; None inherits
        the parent's.
        """
        return ClaudeSession(
            name=name,
            claude_cmd=list(self._claude_cmd),
            protocol=self._protocol,
            control_callback=self._control_callback,
            env_overrides=self._env_overrides,
            cwd=cwd if cwd is not None else self._cwd,
            read_timeout=self._read_timeout,
        )

    def with_model(self, model: str) -> ClaudeSession:
        """Return a new ClaudeSession configured to use a different model."""
        cmd = list(self._claude_cmd)
        # Replace or append --model flag
        if "--model" in cmd:
            idx = cmd.index("--model")
            cmd[idx + 1] = model
        else:
            cmd.extend(["--model", model])
        return ClaudeSession(
            name=self._name,
            claude_cmd=cmd,
            protocol=self._protocol,
            control_callback=self._control_callback,
            env_overrides=self._env_overrides,
            cwd=self._cwd,
            read_timeout=self._read_timeout,
        )

    async def set_permission_mode(self, mode: str) -> None:
        """Change the permission mode on the running Claude CLI subprocess."""
        if self._process is None or not self._process.is_running:
            return
        request_id = f"req_perm_{secrets.token_hex(4)}"
        await self._process.write({
            "type": "control_request",
            "request_id": request_id,
            "request": {
                "subtype": "set_permission_mode",
                "mode": mode,
            },
        })
        # Read the control_response acknowledgement
        while True:
            data = await self._process.read(timeout=self._read_timeout)
            if data is None:
                break
            if data.get("type") == "control_response":
                break

    async def start(self) -> None:
        """Spawn the claude subprocess."""
        _tracer.start_span("session.start", attributes={"session.name": self.name}).end()
        self._process = ClaudeProcess()
        await self._process.start(
            self._claude_cmd,
            _clean_env(self._env_overrides),
            self._cwd or os.getcwd(),
        )
        self._stderr_task = asyncio.create_task(self._forward_stderr())

    async def query(
        self,
        prompt: str,
        block_id: str = "",
        block_name: str = "",
    ) -> QueryResult:
        """Send a prompt and collect the full response.

        Reads messages from the inner claude process until a 'result'
        message.  Forwards assistant/stream_event to the protocol handler.
        Relays control_request via the control_callback.
        """
        with _tracer.start_as_current_span(
            "session.query",
            attributes={"session.name": self.name, "block.id": block_id, "block.name": block_name},
        ) as span:
            assert self._process is not None

            msg = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": prompt,
                },
            }
            await self._process.write(msg)

            response_parts: list[str] = []
            result = QueryResult()

            while True:
                try:
                    data = await self._process.read(timeout=self._read_timeout)
                except ReadTimeoutError:
                    # A CLI that has gone silent past the idle bound is wedged
                    # and will not recover.  Reap it here rather than leaving a
                    # zombie process and three leaked pipes behind; the caller
                    # sees is_running go False and can restart if it wants to.
                    log.warning(
                        "[%s] CLI idle for %ss during query — terminating "
                        "wedged subprocess",
                        self.name,
                        self._read_timeout,
                    )
                    await self.stop()
                    raise
                except ReadInterruptedError:
                    # Interrupt is a deliberate abort (e.g. /stop).  Whoever
                    # asked for it owns the subprocess lifecycle, so leave the
                    # process alone and just unwind.
                    log.info("[%s] query interrupted", self.name)
                    raise
                if data is None:
                    break

                msg_type = data.get("type")

                if msg_type == "system":
                    # Forward system messages (e.g. compacting status,
                    # compact_boundary) so the outer SDK can act on them.
                    if self._protocol:
                        self._protocol.emit_forwarded(
                            data, self.name, block_id, block_name
                        )
                    continue

                if msg_type == "assistant":
                    message = data.get("message", {})
                    content = message.get("content", [])
                    if isinstance(content, str):
                        response_parts.append(content)
                    elif isinstance(content, list):
                        response_parts.extend(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    elif "content" in data and data["content"] != content:
                        flat = data["content"]
                        if isinstance(flat, str):
                            response_parts.append(flat)

                    if self._protocol:
                        self._protocol.emit_forwarded(
                            data, self.name, block_id, block_name
                        )

                elif msg_type == "stream_event":
                    if self._protocol:
                        self._protocol.emit_forwarded(
                            data, self.name, block_id, block_name
                        )

                elif msg_type == "rate_limit_event":
                    continue

                elif msg_type == "control_request":
                    await self._handle_control_request(data)

                elif msg_type == "result":
                    result.response_text = (
                        "\n".join(response_parts)
                        if response_parts
                        else data.get("result", "")
                    )
                    cli_cost = data.get("total_cost_usd", 0.0)
                    result.cost_usd = cli_cost - self._last_cli_cost
                    self._last_cli_cost = cli_cost
                    result.duration_ms = data.get("duration_ms", 0)
                    result.token_usage = self._accumulate_usage(data.get("usage") or {})
                    self._total_cost += result.cost_usd

                    if data.get("session_id"):
                        self._session_id = data["session_id"]
                    break

            if not result.response_text and response_parts:
                result.response_text = "\n".join(response_parts)

            span.set_attributes({
                "session.cost_usd": result.cost_usd,
                "session.duration_ms": result.duration_ms,
            })
            return result

    async def stream_query(
        self,
        prompt: str,
        block_id: str = "",
        block_name: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a prompt and yield raw JSON messages as they arrive.

        Yields each message dict from the inner CLI's stream-json output
        (system, assistant, stream_event, result).  control_request is
        handled transparently and not yielded; rate_limit_event is
        filtered out.  Cost and session_id are updated on the terminal
        'result' message before it is yielded.
        """
        with _tracer.start_as_current_span(
            "session.stream_query",
            attributes={"session.name": self.name, "block.id": block_id, "block.name": block_name},
        ) as span:
            assert self._process is not None

            await self._process.write({
                "type": "user",
                "message": {"role": "user", "content": prompt},
            })

            while True:
                data = await self._process.read(timeout=self._read_timeout)
                if data is None:
                    return

                msg_type = data.get("type")

                if msg_type == "control_request":
                    await self._handle_control_request(data)
                    continue

                if msg_type == "rate_limit_event":
                    continue

                if self._protocol and msg_type in ("system", "assistant", "stream_event"):
                    self._protocol.emit_forwarded(
                        data, self.name, block_id, block_name
                    )

                if msg_type == "result":
                    cli_cost = data.get("total_cost_usd", 0.0)
                    cost = cli_cost - self._last_cli_cost
                    self._last_cli_cost = cli_cost
                    self._total_cost += cost
                    self._accumulate_usage(data.get("usage") or {})
                    if data.get("session_id"):
                        self._session_id = data["session_id"]
                    span.set_attributes({
                        "session.cost_usd": cost,
                        "session.duration_ms": data.get("duration_ms", 0),
                    })
                    yield data
                    return

                yield data

    async def _handle_control_request(self, request: dict[str, Any]) -> None:
        """Relay a control request from inner claude to the client."""
        assert self._process is not None

        if self._control_callback:
            response = await self._control_callback(request)
            await self._process.write(response)
        else:
            # No callback — deny the request
            deny = {
                "type": "control_response",
                "response": {
                    "request_id": request.get("request_id", ""),
                    "allowed": False,
                },
            }
            await self._process.write(deny)

    async def _forward_stderr(self) -> None:
        """Read stderr lines and forward to protocol/log."""
        assert self._process is not None
        while True:
            line = await self._process.read_stderr()
            if line is None:
                break
            if self._protocol:
                self._protocol.emit_stderr(line, self.name)
            else:
                log.debug("[%s stderr] %s", self.name, line)

    async def clear(self) -> None:
        """Clear conversation by restarting the subprocess with a fresh session.

        Cost tracking is preserved across restarts.  Any conversation-continuity
        flags (``--resume``/``--continue``/a pinned ``--session-id``) are
        stripped from the launch command first: a refresh means "do not
        continue", and respawning with ``--resume`` would reload the very
        conversation the refresh is meant to discard.
        """
        _tracer.start_span("session.clear", attributes={"session.name": self.name}).end()
        self._last_cli_cost = 0.0
        self._claude_cmd = _strip_continuity_flags(self._claude_cmd)
        self._session_id = None
        await self.stop()
        await self.start()

    async def stop(self) -> None:
        """Terminate the subprocess."""
        _tracer.start_span("session.stop", attributes={"session.name": self.name}).end()
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None
        if self._process:
            self._process.interrupt()
            await self._process.stop()
            self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_running


# Backwards-compatible alias so existing imports keep working.
Session = ClaudeSession
