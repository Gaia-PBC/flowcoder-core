"""A spawned child's cost is surfaced to the parent flowchart.

Parallel to ``exit_code_variable``: a ``SpawnBlock`` may name a ``cost_variable``
and, after the child is joined at a wait, that parent variable holds the child's
cumulative ``cost_usd``. The value flows child-session ``total_cost`` ->
``ExecutionResult.cost_usd`` (built in ``GraphWalker.run``) -> parent variable
(written in ``_exec_wait``).

These tests pin both halves: that ``run`` copies the session's cost into the
result, and that ``_exec_wait`` writes the joined child's cost into the named
variable -- independently of the ``exit_code_variable`` write beside it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from flowcoder_engine.walker import GraphWalker
from flowcoder_flowchart import (
    Connection,
    EndBlock,
    ExitBlock,
    Flowchart,
    SpawnBlock,
    StartBlock,
    WaitBlock,
)

from tests.conftest import MockProtocol, MockSession

# A distinctive, non-default cost so a passing assertion proves the *child's*
# cost propagated -- not the 0.0 default, and not a value read off the wrong
# session.
_CHILD_COST = 0.4242


class _CostSession(MockSession):
    """A MockSession whose spawned-child clones report a fixed cumulative cost.

    A real session accrues cost as it queries (session.py:
    ``self._total_cost += result.cost_usd``); the base MockSession never does,
    so we stamp a known cost onto every clone. The walker clones the parent
    session per spawn (``self._session.clone(agent_name)``), so the child that
    runs the sub-flowchart reports ``_CHILD_COST`` as its ``total_cost``.
    """

    def clone(self, name: str, cwd: str | None = None) -> MockSession:
        child = super().clone(name, cwd)
        child._total_cost = _CHILD_COST
        return child


def _child_flowchart() -> Flowchart:
    """A trivial start -> end flowchart for the spawned command to run."""
    return Flowchart(
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[Connection(source_id="cs", target_id="ce")],
    )


def _child_flowchart_exit(code: int) -> Flowchart:
    """A child that exits with a specific code, so exit_code is non-default."""
    return Flowchart(
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "cx": ExitBlock(id="cx", name="ChildExit", exit_code=code),
        },
        connections=[Connection(source_id="cs", target_id="cx")],
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


class TestExecutionResultCost:
    async def test_cost_usd_populated_from_session_total_cost(self):
        """``GraphWalker.run`` copies the session's cumulative cost into the result."""
        session = MockSession()
        # Simulate cost accrued during the run (real sessions bump this per query).
        session._total_cost = 0.37
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[Connection(source_id="s", target_id="e")],
        )
        walker = GraphWalker(fc, session, {}, MockProtocol())

        result = await walker.run()

        assert result.status == "completed"
        # Default is 0.0; this proves the run wired session.total_cost through.
        assert result.cost_usd == 0.37

    async def test_cost_usd_defaults_to_zero_when_session_free(self):
        """A run on a zero-cost session reports cost_usd 0.0, not a stale value."""
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[Connection(source_id="s", target_id="e")],
        )
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())

        result = await walker.run()

        assert result.cost_usd == 0.0


class TestCostVariable:
    async def test_cost_variable_receives_child_cost_after_wait(self):
        """A spawn's ``cost_variable`` holds the joined child's ``cost_usd``."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            cost_variable="cost_1",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _CostSession(), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        # KeyError here (pre-fix) proves the variable is only present because
        # _exec_wait wrote it; the value proves it's the child's real cost.
        assert result.variables["cost_1"] == _CHILD_COST

    async def test_cost_variable_absent_leaves_no_spurious_variable(self):
        """No ``cost_variable`` -> nothing is written for cost."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _CostSession(), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        assert not any(
            v == _CHILD_COST for v in result.variables.values()
        ), result.variables

    async def test_cost_and_exit_code_variables_surface_independently(self):
        """Both variables are written from the same joined child, independently."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            cost_variable="cost_1",
            exit_code_variable="rc_1",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _CostSession(), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart_exit(3)
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        assert result.variables["cost_1"] == _CHILD_COST
        assert result.variables["rc_1"] == 3

    async def test_cost_variable_with_join_all_wait(self):
        """An empty (join-all) wait still surfaces each child's cost_variable."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            cost_variable="cost_1",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=[])  # join all pending
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _CostSession(), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        assert result.variables["cost_1"] == _CHILD_COST
