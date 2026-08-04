"""Subprocess manager for the inner Claude CLI.

Spawns claude as a child process with PIPE for stdin and stdout.
Claude CLI with --output-format stream-json writes JSON lines to stdout.
Provides sequential read/write — no event queues, no async generators.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from asyncio.subprocess import PIPE
from typing import Any

log = logging.getLogger(__name__)

# Idle bound for a single read from the inner CLI's stdout: the maximum
# time we wait for the NEXT byte, not the total time a query may take.
# A total-elapsed cap would be wrong here — a legitimate turn can run
# for many minutes.
#
# Sized for false-positive safety rather than fast detection, because the
# two errors are not symmetric.  The failure this guards against is a
# wedged CLI, observed idling for ~2.5 hours; any bound in the minutes
# range catches that effectively instantly, so a smaller value buys
# almost nothing.  A bound that is too SMALL, though, halts a healthy
# flowchart mid-turn, and this value is not reachable by consumers —
# retuning it costs a flowcoder release and a dependency re-pin.
#
# Measured, not assumed: driving a real CLI through a 90s `sleep` tool
# call produced 27 stdout messages with a maximum inter-message gap of
# 2.9s — it keeps streaming while a tool runs, even without
# --include-partial-messages.  So healthy traffic stays far below this
# bound, and the generous value costs nothing in practice.  Re-measure
# before assuming a tighter bound is unsafe; the earlier worry that a
# tool call implies an equally long stdout silence proved false.
QUERY_READ_TIMEOUT = 300.0


class ReadTimeoutError(Exception):
    """CLI subprocess produced no output within the idle timeout."""


class ReadInterruptedError(Exception):
    """CLI subprocess read was aborted by interrupt() (e.g. /stop)."""


def find_claude() -> str:
    """Find the claude CLI binary on PATH.

    Raises FileNotFoundError if not found.
    """
    path = shutil.which("claude")
    if path:
        return path
    raise FileNotFoundError(
        "Could not find 'claude' CLI on PATH. "
        "Install it or pass --claude-path explicitly."
    )


class ClaudeProcess:
    """A single Claude CLI subprocess.

    Claude CLI with --output-format stream-json writes JSON lines to stdout.
    Sequential interface: call read() in a loop to get one JSON message
    at a time.  No background tasks, no queues.
    """

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._interrupt: asyncio.Event = asyncio.Event()

    async def start(self, cmd: list[str], env: dict[str, str], cwd: str) -> None:
        """Spawn the subprocess.

        This is the ONLY place the interrupt latch is cleared — see
        interrupt() for why it is otherwise terminal.  A newly spawned
        process has a clean stdout stream, so inheriting a previous
        run's abort would be wrong.
        """
        self._interrupt.clear()
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            env=env,
            cwd=cwd or None,
            limit=10 * 1024 * 1024,  # 10 MB — Claude stream-json lines can be large
        )

    async def write(self, msg: dict[str, Any]) -> None:
        """Write a JSON message to stdin."""
        assert self._proc is not None
        assert self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self._proc.stdin.drain()

    def interrupt(self) -> None:
        """Abort any in-flight read().  TERMINAL — this process is now done.

        The interrupt latches: it is never auto-cleared, so every later
        read() on this process raises ReadInterruptedError immediately.
        That is deliberate, not an oversight.  Aborting a read mid-turn
        leaves the CLI's stdout desynchronized — the inner CLI may still
        emit the rest of that turn (assistant deltas, its terminal
        'result'), and a subsequent query() would write a new prompt and
        then consume those leftovers as if they answered it, corrupting
        the response text and the cost accounting.  Latching makes that
        unsafe reuse impossible instead of merely unlikely.

        start() is the single defined reset point: it clears the latch
        because a freshly spawned process has a clean stream.  Callers
        that want a usable session after interrupting go through
        ClaudeSession.clear()/stop()+start(), which spawn a new process.
        """
        self._interrupt.set()

    async def read(
        self, timeout: float | None = QUERY_READ_TIMEOUT
    ) -> dict[str, Any] | None:
        """Read one JSON message from stdout.

        Returns None on EOF (process exited).  Skips non-JSON lines.

        *timeout* is an IDLE bound applied per underlying readline — the
        limit on waiting for the next line, not on the total call.  It
        defaults to QUERY_READ_TIMEOUT so an unbounded wait is never the
        accidental default; pass timeout=None to opt out explicitly.

        Raises ReadTimeoutError if no line arrives within *timeout*.
        Raises ReadInterruptedError if interrupt() is called during the wait.
        """
        assert self._proc is not None
        assert self._proc.stdout is not None
        while True:
            line_bytes = await self._read_line(timeout)
            if not line_bytes:
                return None
            line = line_bytes.decode().strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    async def _read_line(self, timeout: float | None) -> bytes:
        """Read one raw line from stdout, racing readline against interrupt.

        Partial-line safety: when the readline task is cancelled (timeout or
        interrupt), bytes already consumed stay in the StreamReader's internal
        buffer — asyncio only drains that buffer once a full separator is
        found.  A resumed read therefore returns the complete line rather
        than a truncated one, so no partial JSON is ever surfaced.
        """
        assert self._proc is not None
        assert self._proc.stdout is not None

        if self._interrupt.is_set():
            raise ReadInterruptedError("CLI read interrupted")

        read_task = asyncio.ensure_future(self._proc.stdout.readline())
        interrupt_task = asyncio.ensure_future(self._interrupt.wait())

        try:
            done, pending = await asyncio.wait(
                {read_task, interrupt_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            # Cancellation of the awaiting coroutine must not orphan the tasks.
            read_task.cancel()
            interrupt_task.cancel()
            raise

        # A line that already arrived wins over a simultaneous interrupt —
        # discarding it here would silently drop a complete CLI message.
        if read_task in done and not read_task.cancelled():
            interrupt_task.cancel()
            return read_task.result()

        for t in pending:
            t.cancel()

        if interrupt_task in done:
            raise ReadInterruptedError("CLI read interrupted")

        raise ReadTimeoutError(
            f"No output from CLI subprocess for {timeout}s (idle timeout)"
        )

    async def read_stderr(self) -> str | None:
        """Read one line from stderr. Returns None on EOF."""
        if not self._proc or not self._proc.stderr:
            return None
        line_bytes = await self._proc.stderr.readline()
        if not line_bytes:
            return None
        return line_bytes.decode().rstrip("\n")

    async def stop(self) -> None:
        """Terminate the subprocess and clean up."""
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (ProcessLookupError, TimeoutError):
                if self._proc.returncode is None:
                    self._proc.kill()
            finally:
                self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None
