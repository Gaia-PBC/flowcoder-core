"""End-to-end tests for the Input block, driven through a real engine subprocess.

Unlike ``tests/integration/test_e2e.py`` — which runs ``GraphWalker`` in-process
with a ``_MockProtocol`` whose ``read_message``/``_inbox`` do not exist — these
launch the engine as a subprocess so the *real* ``ProtocolHandler`` +
``_MessageRouter`` delivery path is exercised.  That path is exactly where the
input-block hang lives, and exactly what an in-process mock cannot catch.

Pre-fix behaviour these assert against: the engine emits ``input_request`` and
then never completes the input block — ``read_message()`` blocks on a
``ProtocolHandler._inbox`` that nothing feeds in engine mode — so both tests
TIME OUT against the current tree.  They pass once the delivery fix (route the
router's queue to the walker) and the read bound land.  A stub ``claude`` is
used, so they cost zero tokens and carry no ``slow`` marker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_harness import EngineHarness, subtype  # noqa: E402


def _write_command(
    command_dir: Path, name: str, *, timeout_seconds: int | None = None
) -> None:
    """Write a ``start -> input -> end`` command to ``<dir>/commands/<name>.json``."""
    input_block: dict = {"id": "i", "type": "input", "name": "Ask"}
    if timeout_seconds is not None:
        input_block["timeout_seconds"] = timeout_seconds
    command = {
        "name": name,
        "description": "start -> input -> end",
        "flowchart": {
            "blocks": {
                "s": {"id": "s", "type": "start", "name": "Start"},
                "i": input_block,
                "e": {"id": "e", "type": "end", "name": "End"},
            },
            "connections": [
                {"source_id": "s", "target_id": "i"},
                {"source_id": "i", "target_id": "e"},
            ],
        },
    }
    cmd_dir = command_dir / "commands"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    (cmd_dir / f"{name}.json").write_text(json.dumps(command))


async def test_input_block_completes_after_response(tmp_path: Path) -> None:
    """The bug this deck is named for: an input block must complete once a
    response is delivered.  Send /inputflow, observe ``input_request``, deliver
    an ``input_response``, and expect the input block to complete and the
    flowchart to finish."""
    _write_command(tmp_path, "inputflow")
    async with EngineHarness(tmp_path) as eng:
        await eng.send_user("/inputflow")

        req = await eng.recv_until(lambda m: subtype(m) == "input_request")
        block_id = req["data"]["block_id"]

        await eng.send(
            {"type": "input_response", "block_id": block_id, "content": "hi"}
        )

        done = await eng.recv_until(
            lambda m: subtype(m) == "block_complete"
            and m["data"]["block_id"] == block_id
        )
        assert done["data"]["success"] is True

        fc = await eng.recv_until(lambda m: subtype(m) == "flowchart_complete")
        assert fc["data"]["status"] == "completed"


async def test_input_block_times_out_when_no_response(tmp_path: Path) -> None:
    """The bound half of the fix: with no response delivered, the input block
    must FAIL within its ``timeout_seconds`` rather than hang forever.  We give
    it a short 2s bound and assert it reports failure well inside the harness
    idle timeout."""
    _write_command(tmp_path, "inputflow", timeout_seconds=2)
    async with EngineHarness(tmp_path) as eng:
        await eng.send_user("/inputflow")

        req = await eng.recv_until(lambda m: subtype(m) == "input_request")
        block_id = req["data"]["block_id"]

        # Deliberately send nothing.  Post-fix, the block times out at ~2s and
        # reports a failed block_complete; pre-fix it hangs and this times out.
        done = await eng.recv_until(
            lambda m: subtype(m) == "block_complete"
            and m["data"]["block_id"] == block_id,
            timeout=12.0,
        )
        assert done["data"]["success"] is False
