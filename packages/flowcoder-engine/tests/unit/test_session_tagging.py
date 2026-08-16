"""Session tagging of lifecycle events + spawn_start/spawn_complete ordering.

The engine emits system events tagged with the emitting walker's session name
(``self._session.name``): block/flowchart lifecycle events carry ``session`` in
their data, and a spawn/wait pair emits ``spawn_start`` when the child task is
created and ``spawn_complete`` when it finishes (completed/failed/cancelled),
with ``session == parent_session`` == the spawning walker's session name.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from flowcoder_engine.walker import GraphWalker
from flowcoder_flowchart import (
    Connection,
    EndBlock,
    Flowchart,
    PromptBlock,
    SpawnBlock,
    StartBlock,
    WaitBlock,
)

from tests.conftest import MockProtocol, MockSession


def _child_flowchart() -> Flowchart:
    """A trivial start -> end flowchart for the spawned command to run."""
    return Flowchart(
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[Connection(source_id="cs", target_id="ce")],
    )


def _spawn_wait_flowchart() -> Flowchart:
    """start -> spawn -> wait -> end."""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "sp": SpawnBlock(
                id="sp",
                name="Spawn",
                agent_name="worker",
                command_name="child-cmd",
            ),
            "w": WaitBlock(id="w", name="Wait", wait_for=["worker"]),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="sp"),
            Connection(source_id="sp", target_id="w"),
            Connection(source_id="w", target_id="e"),
        ],
    )


def _messages_of(protocol: MockProtocol, subtype: str) -> list[dict]:
    return [m for m in protocol.messages if m.get("subtype") == subtype]


class TestSpawnLifecycleEvents:
    async def test_spawn_start_then_complete_in_order(self):
        """A flowchart with a spawn + wait emits spawn_start then spawn_complete,
        each tagged with the emitting (parent) session name."""
        fc = _spawn_wait_flowchart()
        parent = MockSession()  # _name == "mock"
        protocol = MockProtocol()
        walker = GraphWalker(fc, parent, {}, protocol)

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        assert result.status == "completed", result
        starts = _messages_of(protocol, "spawn_start")
        completes = _messages_of(protocol, "spawn_complete")
        assert len(starts) == 1
        assert len(completes) == 1

        start = starts[0]
        assert start["data"]["agent_name"] == "worker"
        assert start["data"]["command_name"] == "child-cmd"
        assert start["data"]["parent_session"] == "mock"
        assert start["data"]["session"] == "mock", "session == parent_session"

        complete = completes[0]
        assert complete["data"]["agent_name"] == "worker"
        assert complete["data"]["status"] == "completed"
        assert complete["data"]["session"] == "mock"

        # spawn_start must precede spawn_complete in the emission stream.
        stream = [
            m.get("subtype")
            for m in protocol.messages
            if m.get("subtype") in ("spawn_start", "spawn_complete")
        ]
        assert stream == ["spawn_start", "spawn_complete"]

    async def test_block_events_carry_walker_session(self):
        """block_start/block_complete are tagged with the emitting walker's session.

        The parent walker emits events for its 4 blocks (start, spawn, wait,
        end) tagged 'mock'; the spawned child walker emits events for its own 2
        blocks tagged with the child session name ('worker')."""
        fc = _spawn_wait_flowchart()
        parent = MockSession()
        protocol = MockProtocol()
        walker = GraphWalker(fc, parent, {}, protocol)

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        assert result.status == "completed", result
        starts = _messages_of(protocol, "block_start")
        completes = _messages_of(protocol, "block_complete")
        # Parent walker: start, spawn, wait, end.  Child walker: child start, child end.
        assert len(starts) == 6
        assert len(completes) == 6
        parent_starts = [m for m in starts if m["data"].get("session") == "mock"]
        child_starts = [m for m in starts if m["data"].get("session") == "worker"]
        assert len(parent_starts) == 4
        assert len(child_starts) == 2
        parent_completes = [m for m in completes if m["data"].get("session") == "mock"]
        child_completes = [m for m in completes if m["data"].get("session") == "worker"]
        assert len(parent_completes) == 4
        assert len(child_completes) == 2

    async def test_spawn_complete_failed_on_child_error(self):
        """A child that fails reports spawn_complete status='failed'."""

        class FailingChildSession(MockSession):
            async def query(self, prompt: str, block_id: str = "", block_name: str = "") -> None:
                raise RuntimeError("child exploded")

            def clone(self, name: str, cwd: str | None = None) -> MockSession:
                child = FailingChildSession(
                    responses=list(self._responses),
                    delay_seconds=self._delay_seconds,
                    session_id=self._session_id,
                )
                child._name = name
                child._clone_cwd = cwd
                return child

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn", agent_name="worker", command_name="child-cmd"
                ),
                "w": WaitBlock(id="w", name="Wait", wait_for=["worker"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )
        protocol = MockProtocol()
        # The child session is a clone of the parent, so the parent must be the
        # failing session type for the spawned child to fail too.
        walker = GraphWalker(fc, FailingChildSession(), {}, protocol)

        mock_cmd = MagicMock()
        mock_cmd.flowchart = Flowchart(
            blocks={
                "cs": StartBlock(id="cs", name="ChildStart"),
                "cp": PromptBlock(id="cp", name="ChildPrompt", prompt="hi"),
                "ce": EndBlock(id="ce", name="ChildEnd"),
            },
            connections=[
                Connection(source_id="cs", target_id="cp"),
                Connection(source_id="cp", target_id="ce"),
            ],
        )
        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        # The wait block fails (child failed) and the flowchart halts, but the
        # spawn_complete for the failed child must still be emitted.
        completes = _messages_of(protocol, "spawn_complete")
        assert len(completes) == 1
        assert completes[0]["data"]["status"] == "failed"
        assert completes[0]["data"]["session"] == "mock"


def _slow_child_flowchart() -> Flowchart:
    """A start -> prompt -> end flowchart whose prompt sleeps (delay_seconds)."""
    return Flowchart(
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "cp": PromptBlock(id="cp", name="ChildPrompt", prompt="hi"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[
            Connection(source_id="cs", target_id="cp"),
            Connection(source_id="cp", target_id="ce"),
        ],
    )


class TestSpawnCompleteExactlyOnce:
    async def test_wait_timeout_emits_exactly_one_failed(self):
        """A wait whose child times out emits exactly one 'failed' spawn_complete
        (from the timeout branch) and no 'cancelled' from cleanup."""
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn", agent_name="worker", command_name="child-cmd"
                ),
                "w": WaitBlock(
                    id="w", name="Wait", wait_for=["worker"], timeout_seconds=1
                ),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )
        protocol = MockProtocol()
        # Child prompt sleeps 60s; the wait times out at 1s.
        walker = GraphWalker(fc, MockSession(delay_seconds=60), {}, protocol)

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _slow_child_flowchart()
        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        assert result.status != "completed"
        completes = _messages_of(protocol, "spawn_complete")
        assert len(completes) == 1, (
            "timeout branch must emit exactly one spawn_complete; "
            f"cleanup must not double-emit (got {len(completes)})"
        )
        assert completes[0]["data"]["status"] == "failed"
        assert completes[0]["data"]["result"] == "timed out after 1s"
        assert completes[0]["data"]["session"] == "mock"
        assert not _messages_of(protocol, "spawn_cancelled")  # no such subtype
        assert all(
            m["data"]["status"] != "cancelled" for m in protocol.messages
            if m.get("subtype") == "spawn_complete"
        )

    async def test_spawn_with_no_wait_emits_exactly_one_cancelled(self):
        """A spawned-but-never-awaited running child is cancelled at cleanup and
        gets exactly one 'cancelled' spawn_complete."""
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn", agent_name="worker", command_name="child-cmd"
                ),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="e"),
            ],
        )
        protocol = MockProtocol()
        # Child prompt sleeps 60s, so it is still running when the parent ends.
        walker = GraphWalker(fc, MockSession(delay_seconds=60), {}, protocol)

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _slow_child_flowchart()
        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        assert result.status == "completed", result
        starts = _messages_of(protocol, "spawn_start")
        completes = _messages_of(protocol, "spawn_complete")
        assert len(starts) == 1
        assert len(completes) == 1, (
            "cleanup must emit exactly one spawn_complete for the "
            f"never-awaited child (got {len(completes)})"
        )
        assert completes[0]["data"]["status"] == "cancelled"
        assert completes[0]["data"]["session"] == "mock"

    async def test_never_awaited_done_child_emits_real_terminal_complete(self):
        """A child that finishes while the parent continues (no covering wait)
        must still get a terminal spawn_complete from cleanup — with its real
        outcome, not a spurious 'cancelled'."""
        class _FastChildSession(MockSession):
            def clone(self, name: str, cwd: str | None = None) -> MockSession:
                child = MockSession(
                    responses=list(self._responses),
                    delay_seconds=0.01,  # child finishes quickly
                    session_id=self._session_id,
                )
                child._name = name
                child._clone_cwd = cwd
                return child

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn", agent_name="worker", command_name="child-cmd"
                ),
                # No wait block: the parent moves straight on; the child runs in
                # the background and finishes before the parent ends.
                "p": PromptBlock(id="p", name="ParentPrompt", prompt="hi"),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="p"),
                Connection(source_id="p", target_id="e"),
            ],
        )
        protocol = MockProtocol()
        # Parent prompt sleeps 0.2s — long enough for the 0.01s child to finish.
        walker = GraphWalker(fc, _FastChildSession(delay_seconds=0.2), {}, protocol)

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _slow_child_flowchart()
        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        assert result.status == "completed", result
        completes = _messages_of(protocol, "spawn_complete")
        assert len(completes) == 1, (
            "cleanup must emit exactly one terminal spawn_complete for the "
            f"done-but-unreported child (got {len(completes)})"
        )
        assert completes[0]["data"]["status"] == "completed"
        assert completes[0]["data"]["agent_name"] == "worker"
        assert completes[0]["data"]["session"] == "mock"
        assert completes[0]["data"]["result"] == "{}"  # child variables, json-dumped
        assert isinstance(completes[0]["data"]["duration_ms"], int)
