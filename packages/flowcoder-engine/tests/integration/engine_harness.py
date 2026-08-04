"""Reusable end-to-end harness for the flowcoder engine.

Launches the engine as a real subprocess — ``python -m flowcoder_engine`` with a
stub ``claude`` — and speaks its stream-json protocol over real pipes.  This is
the capability the in-process ``test_e2e.py`` lacks: those tests hand the walker
a ``_MockProtocol`` whose ``read_message``/``_inbox`` do not exist, so they
structurally cannot exercise (or catch a bug in) the input-delivery path.  Here
the engine drives its *own* ``ProtocolHandler`` + ``_MessageRouter``, so the path
under test is the real one.

Typical use::

    async with EngineHarness(command_dir) as eng:
        await eng.send_user("/inputflow")
        req = await eng.recv_until(lambda m: subtype(m) == "input_request")
        await eng.send({"type": "input_response",
                        "block_id": req["data"]["block_id"], "content": "hi"})
        done = await eng.recv_until(lambda m: subtype(m) == "flowchart_complete")
"""
from __future__ import annotations

import asyncio
import json
import sys
from asyncio.subprocess import PIPE
from collections.abc import Callable
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
STUB_CLAUDE = _HERE / "_stub_claude.py"

# Default idle bound for a single protocol message from the engine.  Sized so a
# *broken* engine (an input block that hangs forever) fails a test in seconds
# rather than hanging CI, while a healthy engine answers far faster.
DEFAULT_TIMEOUT = 15.0

# Match the engine's own inner-pipe limit so a large forwarded line cannot
# overrun asyncio's default 64 KB StreamReader buffer.
_STREAM_LIMIT = 10 * 1024 * 1024


def subtype(msg: dict[str, Any] | None) -> str | None:
    """The system-message subtype, or ``None`` for a non-system message."""
    if msg and msg.get("type") == "system":
        return msg.get("subtype")
    return None


class EngineTimeout(Exception):
    """The engine produced no matching message within the timeout."""


class EngineClosed(Exception):
    """The engine's stdout hit EOF before a matching message arrived."""


class EngineHarness:
    """Drive one engine subprocess over stream-json pipes."""

    def __init__(
        self,
        command_dir: str | Path,
        *,
        extra_args: list[str] | None = None,
    ) -> None:
        self._command_dir = Path(command_dir)
        self._extra_args = list(extra_args or [])
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr: list[str] = []
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        # A tiny launcher guarantees the stub runs under *this* interpreter,
        # independent of PATH or the file's exec bit: ``--claude-path`` takes a
        # single path and the engine appends the -p/stream-json flags to it.
        launcher = self._command_dir / "_claude_launcher.sh"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{STUB_CLAUDE}" "$@"\n'
        )
        launcher.chmod(0o755)

        cmd = [
            sys.executable,
            "-m",
            "flowcoder_engine",
            "--claude-path",
            str(launcher),
            "--search-path",
            str(self._command_dir),
            *self._extra_args,
        ]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            cwd=str(self._command_dir),
            limit=_STREAM_LIMIT,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            self._stderr.append(line.decode(errors="replace").rstrip("\n"))

    async def send(self, msg: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self._proc.stdin.drain()

    async def send_user(self, content: str) -> None:
        await self.send(
            {"type": "user", "message": {"role": "user", "content": content}}
        )

    async def recv(self, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any] | None:
        """Return the next JSON protocol message from the engine's stdout.

        Returns ``None`` on EOF.  Raises :class:`EngineTimeout` if no line
        arrives within *timeout* (an idle bound).  Blank and non-JSON lines are
        skipped.
        """
        assert self._proc is not None and self._proc.stdout is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise EngineTimeout(
                    f"no protocol message within {timeout}s{self._diag()}"
                )
            try:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(), remaining
                )
            except asyncio.TimeoutError as e:
                raise EngineTimeout(
                    f"no protocol message within {timeout}s{self._diag()}"
                ) from e
            if not line:
                return None
            text = line.decode().strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue

    async def recv_until(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Read until *predicate* matches; return the matching message.

        Raises :class:`EngineTimeout` on the deadline, or :class:`EngineClosed`
        if stdout hits EOF first.  Both carry the subtypes seen so far and a
        tail of the engine's stderr, so a failure says *what* the engine did.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        seen: list[str] = []
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise EngineTimeout(
                    f"predicate not met within {timeout}s; saw {seen}{self._diag()}"
                )
            msg = await self.recv(timeout=remaining)
            if msg is None:
                raise EngineClosed(
                    f"engine stdout closed; saw {seen}{self._diag()}"
                )
            seen.append(msg.get("subtype") or msg.get("type") or "?")
            if predicate(msg):
                return msg

    def _diag(self) -> str:
        if not self._stderr:
            return ""
        tail = "\n".join(self._stderr[-10:])
        return f"\n--- engine stderr (tail) ---\n{tail}"

    async def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
            if proc.returncode is None:
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        finally:
            if self._stderr_task is not None:
                self._stderr_task.cancel()
            self._proc = None

    async def __aenter__(self) -> EngineHarness:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
