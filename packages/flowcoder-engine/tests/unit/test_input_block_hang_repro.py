"""Version-agnostic reproduction of the input-block hang.

This module deliberately imports ONLY pre-existing API (ClaudeProcess), so
it collects and runs against both the pre-fix and post-fix trees.  That is
what makes it a real fail-before / pass-after reproduction:

  * pre-fix  — read() waits on stdout forever, the outer bound expires and
               the test fails with "HANG REPRODUCED".
  * post-fix — read() raises ReadTimeoutError (or the interrupt fires) well
               inside the bound and the test passes.

A test module that imports the symbols the fix introduces cannot do this:
it fails on the old tree with a collection-time ImportError, which proves
nothing about an infinite await.  Feature detection via inspect/hasattr is
what keeps the failure signal pinned to the hang itself.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from flowcoder_engine.subprocess import ClaudeProcess

# Long enough that a working implementation never trips it, short enough
# that a hang is caught quickly.
HANG_BOUND = 2.0


class SilentStdout:
    """stdout whose readline() never resolves — the wedged-CLI condition."""

    def __init__(self) -> None:
        self.readline_calls = 0

    def readline(self) -> asyncio.Future[bytes]:
        self.readline_calls += 1
        return asyncio.get_running_loop().create_future()  # never set


class FakeStdin:
    def __init__(self) -> None:
        self.written: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeProc:
    def __init__(self) -> None:
        self.stdout = SilentStdout()
        self.stdin = FakeStdin()
        self.stderr = None
        self.returncode = None

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


def _wedged() -> ClaudeProcess:
    cp = ClaudeProcess()
    cp._proc = FakeProc()
    return cp


def _read_supports_timeout() -> bool:
    return "timeout" in inspect.signature(ClaudeProcess.read).parameters


async def _read_bounded(cp: ClaudeProcess):
    """Read, using the idle bound if this version of the API offers one."""
    if _read_supports_timeout():
        return await cp.read(timeout=0.1)
    return await cp.read()  # pre-fix: no bound exists to pass


class TestHangReproduction:
    async def test_read_on_wedged_cli_must_not_hang(self):
        """THE reproduction: a silent CLI must not wedge read() forever."""
        cp = _wedged()
        try:
            await asyncio.wait_for(_read_bounded(cp), timeout=HANG_BOUND)
        except asyncio.TimeoutError:
            pytest.fail(
                f"HANG REPRODUCED: ClaudeProcess.read() did not return within "
                f"{HANG_BOUND}s against a CLI that never writes to stdout. "
                f"readline() was entered {cp._proc.stdout.readline_calls} time(s) "
                f"and never resolved."
            )
        except Exception as exc:  # noqa: BLE001 — any prompt error is the fix working
            assert type(exc).__name__ == "ReadTimeoutError", (
                f"expected ReadTimeoutError from the idle bound, got {exc!r}"
            )

    async def test_default_read_is_bounded(self):
        """Calling read() with no arguments must still be bounded.

        An unbounded default means any caller that forgets the keyword
        silently reintroduces the hang.
        """
        assert _read_supports_timeout(), (
            "HANG REPRODUCED: ClaudeProcess.read() takes no timeout at all, "
            "so every read on a wedged CLI waits forever."
        )
        default = inspect.signature(ClaudeProcess.read).parameters["timeout"].default
        assert default is not None and default > 0, (
            f"read()'s timeout defaults to {default!r} — an unbounded default "
            f"leaves the hang reachable from any caller that omits it."
        )

    async def test_hanging_read_is_interruptible(self):
        """Timeout alone is not enough — /stop must be able to abort a read."""
        assert hasattr(ClaudeProcess, "interrupt"), (
            "NO INTERRUPT PATH: ClaudeProcess has no interrupt(), so a read "
            "blocked on CLI stdout cannot be aborted; only the idle timeout "
            "could ever end it."
        )

        cp = _wedged()

        async def fire() -> None:
            await asyncio.sleep(0.05)
            cp.interrupt()

        asyncio.create_task(fire())

        # Idle bound deliberately far beyond the outer bound, so ONLY the
        # interrupt can end this read.
        try:
            await asyncio.wait_for(cp.read(timeout=3600.0), timeout=HANG_BOUND)
        except asyncio.TimeoutError:
            pytest.fail(
                f"HANG REPRODUCED: interrupt() did not unblock a read that was "
                f"waiting on silent stdout within {HANG_BOUND}s."
            )
        except Exception as exc:  # noqa: BLE001
            assert type(exc).__name__ == "ReadInterruptedError", (
                f"expected ReadInterruptedError from interrupt(), got {exc!r}"
            )
