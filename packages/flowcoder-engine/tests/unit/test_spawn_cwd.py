"""Per-spawn ``cwd`` isolates a spawned child's working directory.

A ``SpawnBlock`` may set ``cwd`` (templated); ``_exec_spawn`` resolves it,
creates the directory, and passes it to ``clone(name, cwd=...)`` so parallel
siblings can run in separate dirs.  Unset -> None -> the child inherits the
parent's cwd (unchanged behavior).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from flowcoder_engine.session import ClaudeSession
from flowcoder_engine.walker import GraphWalker
from flowcoder_flowchart import (
    Connection,
    EndBlock,
    Flowchart,
    SpawnBlock,
    StartBlock,
    WaitBlock,
)

from tests.conftest import MockProtocol, MockSession


class _CwdRecordingSession(MockSession):
    """A MockSession that records the (name, cwd) of every clone call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clone_cwds: list[tuple[str, str | None]] = []

    def clone(self, name: str, cwd: str | None = None) -> MockSession:
        self.clone_cwds.append((name, cwd))
        return super().clone(name, cwd)


def _child_flowchart() -> Flowchart:
    return Flowchart(
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[Connection(source_id="cs", target_id="ce")],
    )


def _spawn_wait_flowchart(spawn: SpawnBlock, wait: WaitBlock) -> Flowchart:
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "spawn": spawn,
            "wait": wait,
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="spawn"),
            Connection(source_id="spawn", target_id="wait"),
            Connection(source_id="wait", target_id="e"),
        ],
    )


class TestClaudeSessionCloneCwd:
    def test_clone_inherits_parent_cwd_by_default(self):
        parent = ClaudeSession("p", ["claude"], cwd="/orig")
        assert parent.clone("child")._cwd == "/orig"

    def test_clone_cwd_override(self):
        parent = ClaudeSession("p", ["claude"], cwd="/orig")
        assert parent.clone("child", cwd="/new")._cwd == "/new"


class TestSpawnCwd:
    async def test_cwd_is_templated_created_and_passed_to_clone(self, tmp_path):
        target = tmp_path / "cell-1-work"
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            cwd="{{base}}/cell-1-work",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        parent = _CwdRecordingSession()
        walker = GraphWalker(fc, parent, {"base": str(tmp_path)}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        # {{base}} resolved, the dir was created, and the resolved path was
        # handed to clone -- so the child runs isolated on disk.
        assert parent.clone_cwds == [("cell-1", str(target))]
        assert target.is_dir()

    async def test_no_cwd_passes_none_and_creates_nothing(self, tmp_path):
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        parent = _CwdRecordingSession()
        walker = GraphWalker(fc, parent, {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        # Backward-compatible: clone called with cwd=None (inherit parent),
        # and no directory is created.
        assert parent.clone_cwds == [("cell-1", None)]
        assert list(tmp_path.iterdir()) == []
