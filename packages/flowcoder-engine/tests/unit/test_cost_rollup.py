"""Nested spawn costs roll up into the parent session's ``total_cost``.

A waited child's cumulative cost is added to the parent session's
``total_cost`` (via ``BaseSession.add_cost``) *unconditionally* -- not gated on
``cost_variable``. This is what lets a wrapper that spawns a doer (e.g. a
soul-lite ``SPAWN_OPUS_DO`` with no ``cost_variable``) report the doer's cost,
and it composes recursively: each wait level rolls its own children up, so a
top-level runner's cost includes deeply-nested descendants. Each child is
rolled up exactly once (it is popped from ``_spawned_tasks`` at its wait), so
there is no double-count.

This is distinct from (and complementary to) ``cost_variable``: that writes a
child's cost into a named parent *variable* for observability; this mutates the
parent *session's* accounting so cost aggregates upward automatically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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

# A distinctive, non-default cost so a passing assertion proves the *child's*
# cost propagated -- not the 0.0 default, and not a value read off the wrong
# session.
_CHILD_COST = 0.4242


class _ChildCostSession(MockSession):
    """Every spawned clone reports a fixed cumulative cost; the parent starts 0.

    The base ``MockSession`` never accrues cost, so we stamp a known cost onto
    each clone. The walker clones the parent per spawn
    (``self._session.clone(agent_name)``), so each child that runs the
    sub-flowchart reports ``_CHILD_COST`` as its ``total_cost``.
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


def _spawn_wait_flowchart(spawns: list[SpawnBlock], wait: WaitBlock) -> Flowchart:
    """start -> spawn* -> wait -> end, chaining the given spawn blocks in order."""
    blocks: dict[str, object] = {"s": StartBlock(id="s", name="Start")}
    connections: list[Connection] = []
    prev = "s"
    for spawn in spawns:
        blocks[spawn.id] = spawn
        connections.append(Connection(source_id=prev, target_id=spawn.id))
        prev = spawn.id
    blocks["wait"] = wait
    connections.append(Connection(source_id=prev, target_id="wait"))
    blocks["e"] = EndBlock(id="e", name="End")
    connections.append(Connection(source_id="wait", target_id="e"))
    return Flowchart(blocks=blocks, connections=connections)


def test_add_cost_accumulates_on_session():
    """``BaseSession.add_cost`` adds to the running total (the single mutator)."""
    s = MockSession()
    assert s.total_cost == 0.0
    s.add_cost(0.10)
    s.add_cost(0.05)
    assert s.total_cost == pytest.approx(0.15)


class TestNestedCostRollup:
    async def test_waited_child_cost_rolls_into_parent(self):
        """A waited child's cost is added to the parent session's total_cost,
        with NO cost_variable set (rollup is unconditional)."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            # deliberately no cost_variable
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart([spawn], wait)
        parent = _ChildCostSession()
        walker = GraphWalker(fc, parent, {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        # Parent started at 0; the waited child's cost is now in the parent
        # total (fail-first: without the _exec_wait rollup this stays 0.0).
        assert parent.total_cost == pytest.approx(_CHILD_COST)
        # ...and it therefore surfaces in the top-level ExecutionResult.cost_usd.
        assert result.cost_usd == pytest.approx(_CHILD_COST)

    async def test_multiple_children_accumulate_once_each(self):
        """Two waited children each add their cost exactly once (no double-count)."""
        s1 = SpawnBlock(
            id="s1", name="S1", agent_name="cell-1", command_name="child-cmd"
        )
        s2 = SpawnBlock(
            id="s2", name="S2", agent_name="cell-2", command_name="child-cmd"
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=[])  # join all pending
        fc = _spawn_wait_flowchart([s1, s2], wait)
        parent = _ChildCostSession()
        walker = GraphWalker(fc, parent, {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        assert parent.total_cost == pytest.approx(2 * _CHILD_COST)

    async def test_rollup_coexists_with_cost_variable(self):
        """The opt-in cost_variable still surfaces the child's cost, and the
        parent total is rolled up too -- the two mechanisms are independent."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            cost_variable="cost_1",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart([spawn], wait)
        parent = _ChildCostSession()
        walker = GraphWalker(fc, parent, {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        # Observability path unchanged:
        assert result.variables["cost_1"] == pytest.approx(_CHILD_COST)
        # Accounting path rolled up:
        assert parent.total_cost == pytest.approx(_CHILD_COST)
