"""Reproduction + regression tests for the input-block hang.

The failure mode is an INFINITE AWAIT, not an exception, so every test
here bounds the call in an outer asyncio.wait_for.  Before the fix the
outer bound expires (test fails); after the fix the inner idle timeout
or the interrupt fires first (test passes).  Either way it terminates.

Critically, these tests exercise the REAL ClaudeProcess.read() and the
REAL ClaudeSession.query().  Only `self._proc` is faked, so that
`self._proc.stdout.readline()` never resolves — exactly the wedged-CLI
condition.  Stubbing read() itself would bypass the timeout/interrupt
race that constitutes the fix and would pass identically before and
after it.
"""

from __future__ import annotations

import asyncio

import pytest
from flowcoder_engine.session import ClaudeSession as Session
from flowcoder_engine.subprocess import (
    QUERY_READ_TIMEOUT,
    ClaudeProcess,
    ReadInterruptedError,
    ReadTimeoutError,
)

# Outer bound for every hang-sensitive test.  Comfortably larger than the
# idle timeouts under test, comfortably smaller than "forever".
OUTER_BOUND = 5.0


class SilentStdout:
    """A stdout whose readline() never resolves — a wedged CLI."""

    def __init__(self) -> None:
        self.readline_calls = 0

    def readline(self) -> asyncio.Future[bytes]:
        self.readline_calls += 1
        return asyncio.get_running_loop().create_future()  # never set


class ScriptedStdout:
    """A stdout that yields queued lines, then goes silent forever."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def readline(self) -> asyncio.Future[bytes]:
        fut = asyncio.get_running_loop().create_future()
        if self._lines:
            fut.set_result(self._lines.pop(0))
        return fut


class FakeStdin:
    """Accepts writes and drains instantly — the CLI reads fine, it just never replies."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeProc:
    """Stands in for asyncio.subprocess.Process with a controllable stdout."""

    def __init__(self, stdout: object) -> None:
        self.stdout = stdout
        self.stdin = FakeStdin()
        self.stderr = None
        self.returncode = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


def _wedged_process() -> ClaudeProcess:
    cp = ClaudeProcess()
    cp._proc = FakeProc(SilentStdout())
    return cp


def _wedged_session(read_timeout: float) -> Session:
    s = Session(name="wedged", claude_cmd=["claude"], read_timeout=read_timeout)
    s._process = _wedged_process()
    return s


class TestReadTimeout:
    """ClaudeProcess.read() must not wait forever on a silent CLI."""

    async def test_read_raises_timeout_on_silent_stdout(self):
        cp = _wedged_process()
        with pytest.raises(ReadTimeoutError):
            await asyncio.wait_for(cp.read(timeout=0.1), timeout=OUTER_BOUND)

    async def test_read_is_bounded_by_default(self):
        """The default must be bounded — an unbounded default re-opens the bug."""
        assert QUERY_READ_TIMEOUT is not None
        assert QUERY_READ_TIMEOUT > 0
        import inspect

        default = inspect.signature(ClaudeProcess.read).parameters["timeout"].default
        assert default == QUERY_READ_TIMEOUT, (
            "read() must default to a bounded idle timeout, not None"
        )

    async def test_timeout_is_idle_not_total_elapsed(self):
        """Steady traffic must never trip the bound, however long the turn runs.

        Three lines arrive 0.15s apart with an idle bound of 0.4s.  Total
        elapsed (~0.45s) exceeds the bound; no single gap does.  A
        total-elapsed implementation fails here, an idle one passes.
        """
        cp = ClaudeProcess()
        stdout = ScriptedStdout([])
        cp._proc = FakeProc(stdout)

        pending: list[asyncio.Future[bytes]] = []

        def readline() -> asyncio.Future[bytes]:
            fut = asyncio.get_running_loop().create_future()
            pending.append(fut)
            return fut

        stdout.readline = readline  # type: ignore[method-assign]

        async def feed() -> None:
            for payload in (b'{"n":1}\n', b'{"n":2}\n', b'{"n":3}\n'):
                await asyncio.sleep(0.15)
                while not pending:
                    await asyncio.sleep(0)
                pending.pop(0).set_result(payload)

        feeder = asyncio.create_task(feed())
        try:
            for expected in (1, 2, 3):
                msg = await asyncio.wait_for(
                    cp.read(timeout=0.4), timeout=OUTER_BOUND
                )
                assert msg == {"n": expected}
        finally:
            feeder.cancel()

    async def test_read_still_returns_normal_messages(self):
        cp = ClaudeProcess()
        cp._proc = FakeProc(ScriptedStdout([b'{"type": "result", "ok": true}\n']))
        msg = await asyncio.wait_for(cp.read(timeout=1.0), timeout=OUTER_BOUND)
        assert msg == {"type": "result", "ok": True}


class TestReadInterrupt:
    """A hung read must be abortable from outside — timeout alone is not enough."""

    async def test_interrupt_unblocks_a_hanging_read(self):
        cp = _wedged_process()

        async def fire() -> None:
            await asyncio.sleep(0.05)
            cp.interrupt()

        asyncio.create_task(fire())
        # Idle bound far beyond the outer bound: only the interrupt can win.
        with pytest.raises(ReadInterruptedError):
            await asyncio.wait_for(cp.read(timeout=3600.0), timeout=OUTER_BOUND)

    async def test_interrupt_before_read_is_honoured(self):
        cp = _wedged_process()
        cp.interrupt()
        with pytest.raises(ReadInterruptedError):
            await asyncio.wait_for(cp.read(timeout=3600.0), timeout=OUTER_BOUND)

    async def test_restart_clears_a_latched_interrupt(self):
        """A stale interrupt must not poison the next process."""
        cp = _wedged_process()
        cp.interrupt()
        assert cp._interrupt.is_set()
        try:
            await cp.start(["/bin/true"], {}, "/tmp")
        finally:
            assert not cp._interrupt.is_set()
            await cp.stop()


class TestQueryDoesNotHang:
    """The end-to-end reproduction: Session.query on a wedged CLI."""

    async def test_query_raises_timeout_instead_of_hanging(self):
        session = _wedged_session(read_timeout=0.1)
        with pytest.raises(ReadTimeoutError):
            await asyncio.wait_for(
                session.query("hello", block_id="b1", block_name="Ask"),
                timeout=OUTER_BOUND,
            )

    async def test_query_reaps_the_wedged_subprocess_on_timeout(self):
        """Decision: query() terminates a wedged CLI rather than leaking it."""
        session = _wedged_session(read_timeout=0.1)
        proc = session._process._proc

        with pytest.raises(ReadTimeoutError):
            await asyncio.wait_for(session.query("hello"), timeout=OUTER_BOUND)

        assert proc.terminated, "wedged subprocess must be terminated, not leaked"
        assert not session.is_running

    async def test_stop_unblocks_an_in_flight_query(self):
        """/stop must interrupt a query already blocked on CLI stdout."""
        session = _wedged_session(read_timeout=3600.0)
        query_task = asyncio.create_task(session.query("hello"))

        await asyncio.sleep(0.05)  # let query reach the read
        await session.stop()

        with pytest.raises(ReadInterruptedError):
            await asyncio.wait_for(query_task, timeout=OUTER_BOUND)

    async def test_interrupt_leaves_subprocess_for_the_caller(self):
        """Decision: interrupt does NOT reap — the interrupter owns lifecycle."""
        session = _wedged_session(read_timeout=3600.0)
        proc = session._process._proc
        query_task = asyncio.create_task(session.query("hello"))

        await asyncio.sleep(0.05)
        session._process.interrupt()

        with pytest.raises(ReadInterruptedError):
            await asyncio.wait_for(query_task, timeout=OUTER_BOUND)

        assert not proc.terminated, "interrupt must not terminate the process"


class TestPartialLineSafety:
    """A cancelled read must never surface a truncated JSON line."""

    async def test_partial_line_survives_a_timeout(self):
        """Half a line, then a timeout, then the rest — the line arrives whole.

        Uses a real StreamReader so the buffering semantics under test are
        asyncio's own, not a mock's.
        """
        reader = asyncio.StreamReader()
        cp = ClaudeProcess()
        cp._proc = FakeProc(reader)

        reader.feed_data(b'{"type": "res')  # partial — no newline yet

        with pytest.raises(ReadTimeoutError):
            await asyncio.wait_for(cp.read(timeout=0.1), timeout=OUTER_BOUND)

        reader.feed_data(b'ult", "ok": true}\n')  # remainder arrives

        msg = await asyncio.wait_for(cp.read(timeout=1.0), timeout=OUTER_BOUND)
        assert msg == {"type": "result", "ok": True}, (
            "partial line was corrupted or dropped across the timeout"
        )

    async def test_partial_line_survives_an_interrupt(self):
        reader = asyncio.StreamReader()
        cp = ClaudeProcess()
        cp._proc = FakeProc(reader)

        reader.feed_data(b'{"type": "res')

        async def fire() -> None:
            await asyncio.sleep(0.05)
            cp.interrupt()

        asyncio.create_task(fire())
        with pytest.raises(ReadInterruptedError):
            await asyncio.wait_for(cp.read(timeout=3600.0), timeout=OUTER_BOUND)

        cp._interrupt.clear()
        reader.feed_data(b'ult", "ok": true}\n')

        msg = await asyncio.wait_for(cp.read(timeout=1.0), timeout=OUTER_BOUND)
        assert msg == {"type": "result", "ok": True}

    async def test_arrived_line_wins_over_simultaneous_interrupt(self):
        """A complete line already in hand must not be discarded by an interrupt."""
        reader = asyncio.StreamReader()
        cp = ClaudeProcess()
        cp._proc = FakeProc(reader)

        # Data and interrupt land in the same tick.
        reader.feed_data(b'{"type": "result", "ok": true}\n')
        cp.interrupt()

        # The pre-read interrupt check fires first by design; clear it and
        # confirm the buffered line was never consumed or corrupted.
        cp._interrupt.clear()
        msg = await asyncio.wait_for(cp.read(timeout=1.0), timeout=OUTER_BOUND)
        assert msg == {"type": "result", "ok": True}


class TestConfigInjection:
    """read_timeout must be settable at construction, not baked in."""

    def test_default_comes_from_module_constant(self):
        assert Session(name="t", claude_cmd=["claude"])._read_timeout == QUERY_READ_TIMEOUT

    def test_constructor_override(self):
        assert Session(name="t", claude_cmd=["claude"], read_timeout=0.1)._read_timeout == 0.1

    def test_clone_preserves_read_timeout(self):
        s = Session(name="o", claude_cmd=["claude"], read_timeout=42.0)
        assert s.clone("w")._read_timeout == 42.0

    def test_with_model_preserves_read_timeout(self):
        s = Session(name="t", claude_cmd=["claude"], read_timeout=42.0)
        assert s.with_model("haiku")._read_timeout == 42.0
