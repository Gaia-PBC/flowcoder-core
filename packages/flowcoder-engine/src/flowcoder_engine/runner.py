"""``flowcoder`` — run a flowchart command from a terminal.

The other entry point in this package, ``flowcoder-engine``, is a stream-json
proxy: it reads JSON messages on stdin and writes JSON on stdout, so it is meant
to be driven by a host framework rather than by a person.  This module is the
human-facing front end.  It resolves a command, runs its flowchart in-process
against a real Claude CLI session, and renders the engine's protocol events as
terminal output instead of JSON lines.

Usage::

    flowcoder mycommand "some argument"
    flowcoder --search-path ./commands --model sonnet mycommand
    flowcoder --list

The runner reuses the engine's own pieces — ``resolve_command``, ``GraphWalker``,
``ClaudeSession``, ``build_inner_claude_cmd`` — so a flowchart behaves identically
here and under the proxy.  The only thing it replaces is the I/O surface:
``TerminalProtocol`` subclasses ``ProtocolHandler`` and overrides ``emit`` to
render, and stdin carries plain lines (answers to input blocks and permission
prompts) rather than protocol messages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import stat
import sys
from pathlib import Path
from typing import Any, TextIO

from flowcoder_flowchart import load_command, validate

from .cli import (
    add_claude_args,
    add_engine_args,
    build_inner_claude_cmd,
    build_inner_env,
    build_variables,
)
from .protocol import ProtocolHandler
from .resolver import CommandNotFoundError, resolve_command
from .session import BaseSession, ClaudeSession
from .session_factory import SessionFactory
from .subprocess import find_claude
from .walker import ExecutionError, GraphWalker

# FlowCoder flowcharts are automation: they run unattended, and every prompt
# block that gets a permission dialog it cannot answer stalls the whole run.
# The GUI ran its sessions with bypassPermissions for that reason, so the CLI
# keeps the same default.  Pass --permission-mode explicitly to change it; the
# runner then answers Claude's permission requests by asking on the terminal.
DEFAULT_PERMISSION_MODE = "bypassPermissions"


# ── terminal styling ──────────────────────────────────────────────────


class Style:
    """ANSI styling that turns itself off when the output is not a terminal."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


def _color_enabled(stream: TextIO, no_color: bool) -> bool:
    """Colour unless disabled explicitly, by NO_COLOR, or by a non-tty target."""
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _summarize_tool_input(tool_input: Any) -> str:
    """One short line describing a tool call's input, for the progress log."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "url", "prompt", "description"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\n", " ")
    return ", ".join(sorted(tool_input)) if tool_input else ""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ── protocol handler that renders instead of emitting JSON ────────────


class TerminalProtocol(ProtocolHandler):
    """Renders engine protocol events as terminal output.

    Every ``emit_*`` helper on ``ProtocolHandler`` funnels through ``emit``, so
    overriding that one method is enough to redirect the whole event stream.

    Stdin is read as plain text, not as protocol messages: ``start`` attaches a
    line reader to the event loop (no worker thread, so a run that ends while a
    prompt is still pending shuts down instead of hanging on a blocked
    ``input()``), and input blocks are answered by pushing a synthesized
    ``input_response`` into the inherited inbox the walker reads from.
    """

    def __init__(
        self,
        style: Style,
        stream: TextIO,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self._style = style
        self._stream = stream
        self._verbose = verbose
        self._stdin: asyncio.StreamReader | None = None
        self._direct_stdin = False
        self._pending: set[asyncio.Task[None]] = set()

    # -- output --------------------------------------------------------

    def write(self, text: str = "") -> None:
        self._stream.write(text + "\n")
        self._stream.flush()

    def log(self, message: str) -> None:
        """Engine chatter — only shown with --verbose (stderr, as upstream)."""
        if self._verbose:
            sys.stderr.write(self._style.dim(f"[flowcoder] {message}") + "\n")
            sys.stderr.flush()

    def emit(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == "system":
            self._render_system(msg.get("subtype", ""), msg.get("data") or {})
        elif self._verbose:
            self.write(self._style.dim(_truncate(json.dumps(msg), 200)))

    def _render_system(self, subtype: str, data: dict[str, Any]) -> None:
        style = self._style
        if subtype == "flowchart_start":
            args = data.get("args") or ""
            head = f"/{data.get('command', '')}" + (f" {args}" if args else "")
            self.write(style.bold(head))
            self.write(style.dim(f"{data.get('block_count', 0)} blocks"))
        elif subtype == "block_start":
            name = data.get("block_name") or data.get("block_id", "")
            self.write("")
            self.write(f"{style.cyan('▶')} {style.bold(str(name))} "
                       f"{style.dim(str(data.get('block_type', '')))}")
        elif subtype == "block_complete":
            if not data.get("success", True):
                self.write(f"{style.red('✗')} {data.get('block_name', '')}")
            elif self._verbose:
                self.write(f"{style.green('✓')} {data.get('block_name', '')}")
        elif subtype == "block_timeout":
            self.write(style.yellow(
                f"⏱ {data.get('block_name', '')} timed out after "
                f"{data.get('timeout_seconds', 0)}s"
            ))
        elif subtype == "input_request":
            self._schedule_input(data)
        elif subtype == "session_message":
            self._render_inner(data)
        elif subtype == "stderr":
            if self._verbose:
                self.write(style.dim(f"  {data.get('line', '')}"))
        elif subtype != "flowchart_complete" and self._verbose:
            # flowchart_complete is skipped: the runner prints its own summary
            # from the ExecutionResult, which carries the same numbers.
            self.write(style.dim(f"{subtype}: {_truncate(json.dumps(data), 160)}"))

    def _render_inner(self, data: dict[str, Any]) -> None:
        """Render one message forwarded from an inner Claude session."""
        style = self._style
        message = data.get("message") or {}
        if message.get("type") != "assistant":
            if self._verbose:
                self.write(style.dim(f"  {_truncate(json.dumps(message), 160)}"))
            return

        content = message.get("message", {}).get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = str(part.get("text", "")).strip()
                if text:
                    self.write(text)
            elif part.get("type") == "tool_use":
                summary = _truncate(_summarize_tool_input(part.get("input")), 80)
                label = f"  {part.get('name', 'tool')}"
                self.write(style.dim(f"{label}({summary})" if summary else label))

    # -- stdin ---------------------------------------------------------

    async def start(self) -> None:
        """Attach a line reader to stdin, if stdin can carry one.

        Deliberately *not* ``super().start()``: that parses stdin as JSON
        protocol messages, whereas here stdin is a person typing.

        Only a tty, pipe or socket is handed to ``connect_read_pipe`` — the
        selector cannot register a regular file or ``/dev/null``, and it fails
        asynchronously inside a loop callback where no ``try`` around this call
        can catch it.  Those stdin kinds never block on read, so they are read
        directly at prompt time instead.
        """
        try:
            mode = os.fstat(sys.stdin.fileno()).st_mode
            pollable = stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or sys.stdin.isatty()
        except (OSError, ValueError, AttributeError):
            return  # no usable stdin at all; prompts answer empty
        if not pollable:
            self._direct_stdin = True
            return

        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        try:
            await loop.connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
            )
        except (OSError, ValueError):
            self._direct_stdin = True
            return
        self._stdin = reader

    async def read_line(self, prompt: str) -> str:
        """Print a prompt and read one line of user input ("" on EOF)."""
        self._stream.write(prompt)
        self._stream.flush()
        if self._stdin is not None:
            raw = await self._stdin.readline()
            line = raw.decode(errors="replace")
        elif self._direct_stdin:
            line = sys.stdin.readline()
        else:
            line = ""
        if not line:
            self._stream.write("\n")
            self._stream.flush()
            return ""
        return line.strip()

    def _schedule_input(self, data: dict[str, Any]) -> None:
        """Answer an input block without blocking the walker's event loop."""
        block_id = data.get("block_id", "")
        label = data.get("block_name") or "input"

        async def _ask() -> None:
            text = await self.read_line(self._style.cyan(f"{label} > "))
            self.push_message(
                {"type": "input_response", "block_id": block_id, "content": text}
            )

        task = asyncio.create_task(_ask())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def stop(self) -> None:
        """Cancel any prompt still waiting for a line."""
        for task in list(self._pending):
            task.cancel()
        for task in list(self._pending):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._pending.clear()


# ── tool permissions ──────────────────────────────────────────────────


class PermissionPrompter:
    """Answers inner Claude's ``can_use_tool`` requests from the terminal.

    Only reachable when the user overrides the ``bypassPermissions`` default —
    under that default Claude never asks.  Without a callback the session denies
    every request silently (see ``ClaudeSession._handle_control_request``), which
    is why this exists rather than leaving the callback unset.
    """

    def __init__(self, protocol: TerminalProtocol, style: Style, assume_yes: bool) -> None:
        self._protocol = protocol
        self._style = style
        self._assume_yes = assume_yes
        self._always: set[str] = set()

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id", "")
        body = request.get("request") or {}
        if body.get("subtype") != "can_use_tool":
            # Anything else (hooks, MCP relays) has no terminal equivalent;
            # acknowledge it so the inner CLI is never left waiting.
            return self._response(request_id, {})

        tool = str(body.get("tool_name", "tool"))
        tool_input = body.get("input") or {}
        if self._assume_yes or tool in self._always:
            return self._response(request_id, {"behavior": "allow", "updatedInput": tool_input})

        summary = _truncate(_summarize_tool_input(tool_input), 100)
        self._protocol.write(
            self._style.yellow(f"⚠ {tool}") + (f" {self._style.dim(summary)}" if summary else "")
        )
        answer = (await self._protocol.read_line(
            self._style.yellow("  allow? [y]es / [n]o / [a]lways: ")
        )).strip().lower()

        if answer.startswith("a"):
            self._always.add(tool)
        elif not answer.startswith("y"):
            return self._response(
                request_id, {"behavior": "deny", "message": "Denied by the user"}
            )
        return self._response(request_id, {"behavior": "allow", "updatedInput": tool_input})

    @staticmethod
    def _response(request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": payload,
            },
        }


# ── argument parsing ──────────────────────────────────────────────────


def parse_runner_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``flowcoder`` arguments.

    Flags come before the command name; everything after it belongs to the
    flowchart, so ``flowcoder run-tests --verbose`` passes ``--verbose`` to the
    command rather than to the runner.
    """
    parser = argparse.ArgumentParser(
        prog="flowcoder",
        description="Run a FlowCoder flowchart command in the terminal",
        epilog="Flags must precede the command name; everything after it is "
               "passed to the flowchart as arguments.",
    )
    add_engine_args(parser)
    add_claude_args(parser)
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_commands",
        help="List the commands that can be resolved and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the flowchart's final variables as JSON on stdout "
             "(progress then goes to stderr)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-allow tool permission requests instead of asking "
             "(only relevant with a non-default --permission-mode)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour in the output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show engine logs, per-block completions and raw protocol events",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Command to run, with or without a leading slash (e.g. deploy or /deploy)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the flowchart ($1, $2, ... and named arguments)",
    )

    args = parser.parse_args(argv)
    # build_inner_claude_cmd appends these raw Claude flags last; the runner has
    # no passthrough of its own because REMAINDER hands everything after the
    # command name to the flowchart.
    args.passthrough = []
    if not args.permission_mode:
        args.permission_mode = DEFAULT_PERMISSION_MODE
    if not args.command and not args.list_commands:
        parser.error("a command name is required (or use --list)")
    return args


def _command_dirs(search_paths: list[str] | None) -> list[Path]:
    """The directories ``resolve_command`` looks in, in the same order."""
    dirs = [Path.cwd() / "commands", Path.cwd()]
    for sp in search_paths or []:
        dirs.extend([Path(sp), Path(sp) / "commands"])
    dirs.append(Path.home() / ".flowcoder" / "commands")
    return list(dict.fromkeys(dirs))


def list_commands(search_paths: list[str] | None, style: Style, stream: TextIO) -> int:
    """Print every resolvable command name, nearest search path first."""
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for directory in _command_dirs(search_paths):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.stem in seen:
                continue
            try:
                command = load_command(path)
            except Exception:
                continue  # not a command file — resolve_command would fail too
            seen.add(path.stem)
            rows.append((path.stem, command.description or ""))

    if not rows:
        stream.write("No commands found. Use --search-path to point at a commands directory.\n")
        return 1
    width = max(len(name) for name, _ in rows)
    for name, description in rows:
        line = f"  {style.bold(name.ljust(width))}"
        if description:
            line += f"  {style.dim(_truncate(description, 70))}"
        stream.write(line + "\n")
    return 0


# ── execution ─────────────────────────────────────────────────────────


def build_runner_claude_cmd(args: argparse.Namespace, claude_path: str) -> list[str]:
    """The engine's inner-Claude command, plus what the terminal front end needs.

    Claude only routes permission requests over the control protocol when it has
    been given a permission-prompt tool.  Without ``--permission-prompt-tool``
    it silently refuses anything not auto-approved: the model answers that it
    "needs permission", the block completes having done nothing, and no request
    ever reaches ``PermissionPrompter``.  Verified against the real CLI — same
    prompt, file written only with the flag.

    Under the default ``bypassPermissions`` nothing is ever asked, so the flag
    is added only when the user has chosen a mode that can prompt.
    """
    cmd = build_inner_claude_cmd(args, claude_path)
    if args.permission_mode != DEFAULT_PERMISSION_MODE:
        cmd = [*cmd, "--permission-prompt-tool", "stdio"]
    return cmd


async def run(args: argparse.Namespace) -> int:
    """Resolve and execute one command.  Returns the process exit code."""
    # In --json mode stdout carries the result document only, so all rendering
    # is diverted to stderr and the two never interleave.
    render_to = sys.stderr if args.json else sys.stdout
    style = Style(_color_enabled(render_to, args.no_color))

    if args.list_commands:
        return list_commands(args.search_paths, style, render_to)

    try:
        claude_path = args.claude_path or find_claude()
    except FileNotFoundError as exc:
        print(style.red(str(exc)), file=sys.stderr)
        return 1

    name = args.command.lstrip("/")
    try:
        command = resolve_command(name, args.search_paths)
    except CommandNotFoundError as exc:
        print(style.red(str(exc)), file=sys.stderr)
        return 1

    validation = validate(command.flowchart)
    if not validation.valid:
        print(style.red(f"Flowchart '{name}' is invalid:"), file=sys.stderr)
        for error in validation.errors:
            print(style.red(f"  - {error}"), file=sys.stderr)
        return 1

    # shlex.join round-trips cleanly through build_variables' shlex.split, so a
    # quoted argument stays one argument.
    args_string = shlex.join(args.args)
    try:
        variables = build_variables(args_string, command.arguments)
    except ValueError as exc:
        print(style.red(str(exc)), file=sys.stderr)
        return 1

    protocol = TerminalProtocol(style=style, stream=render_to, verbose=args.verbose)
    await protocol.start()

    claude_cmd = build_runner_claude_cmd(args, claude_path)
    inner_env = build_inner_env(args)
    cwd = args.cwd or None
    permissions = PermissionPrompter(protocol, style, assume_yes=args.yes)

    def make_session(session_name: str, model: str | None) -> BaseSession:
        cmd = [*claude_cmd, "--model", model] if model else list(claude_cmd)
        return ClaudeSession(
            name=session_name,
            claude_cmd=cmd,
            protocol=protocol,
            control_callback=permissions,
            env_overrides=inner_env,
            cwd=cwd,
        )

    # Spawn blocks create their children through the factory; both backends map
    # to a Claude session, as in the engine (codex routes via env vars).
    factory = SessionFactory()
    factory.register("claude", make_session)
    factory.register("codex", make_session)

    session = make_session("main", None)
    await session.start()

    walker = GraphWalker(
        command.flowchart,
        session,
        variables,
        protocol,
        max_blocks=args.max_blocks,
        search_paths=args.search_paths or [],
        session_factory=factory,
    )

    protocol.emit_flowchart_start(name, args_string, len(command.flowchart.blocks))
    try:
        result = await walker.run()
    except ExecutionError as exc:
        print(style.red(f"Execution failed: {exc}"), file=sys.stderr)
        return 1
    finally:
        await protocol.stop()
        await session.stop()

    _print_summary(result, protocol, style)
    if args.json:
        json.dump(result.variables, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")

    if result.status == "exited":
        return result.exit_code
    return 0 if result.status == "completed" else 1


def _print_summary(result: Any, protocol: TerminalProtocol, style: Style) -> None:
    """One closing line: how it ended, and what it took."""
    label = {
        "completed": style.green("completed"),
        "halted": style.red("halted"),
        "error": style.red("error"),
        "exited": style.yellow(f"exited ({result.exit_code})"),
    }.get(result.status, result.status)
    stats = (
        f"· {len(result.log)} blocks · {result.duration_ms / 1000:.1f}s "
        f"· ${result.cost_usd:.4f}"
    )
    protocol.write("")
    protocol.write(f"{label} {style.dim(stats)}")


def main() -> None:
    """Console-script entry point for ``flowcoder``."""
    args = parse_runner_args()
    try:
        code = asyncio.run(run(args))
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
