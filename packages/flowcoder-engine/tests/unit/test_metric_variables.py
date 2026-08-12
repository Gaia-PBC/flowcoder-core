"""A spawned child's runtime and token breakdown are surfaced to the parent.

Parallel to ``cost_variable``/``exit_code_variable`` (see test_cost_variable.py):
a ``SpawnBlock`` may name a ``duration_variable`` and four token variables
(``input_tokens_variable``, ``output_tokens_variable``,
``cache_creation_tokens_variable``, ``cache_read_tokens_variable``).  After the
child is joined at a wait, those parent variables hold the child's metrics.

The values flow child-session/child-walker -> ``ExecutionResult`` (built in
``GraphWalker.run``) -> parent variable (written in ``_exec_wait``).  These
tests pin both halves for the new metrics, independently of the cost/exit
writes beside them.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from flowcoder_engine.session import TokenUsage
from flowcoder_engine.walker import GraphWalker
from flowcoder_flowchart import (
    Connection,
    EndBlock,
    ExitBlock,
    Flowchart,
    PromptBlock,
    SpawnBlock,
    StartBlock,
    WaitBlock,
)

from tests.conftest import MockProtocol, MockSession

# Distinctive, non-default metrics so a passing assertion proves the *child's*
# values propagated -- not zero defaults, and not a value read off the wrong
# session.
_CHILD_COST = 0.55
_CHILD_TOKENS = TokenUsage(
    input_tokens=101,
    output_tokens=202,
    cache_creation_tokens=303,
    cache_read_tokens=404,
)


class _MetricSession(MockSession):
    """A MockSession whose spawned-child clones report fixed cost + token usage.

    A real session accrues these as it queries; the base MockSession never does,
    so we stamp known values onto every clone.  The walker clones the parent per
    spawn (``self._session.clone(agent_name)``), so the child that runs the
    sub-flowchart reports ``_CHILD_COST``/``_CHILD_TOKENS``.
    """

    def clone(self, name: str, cwd: str | None = None) -> MockSession:
        child = super().clone(name, cwd)
        child._total_cost = _CHILD_COST
        child._token_usage = _CHILD_TOKENS
        return child


def _trivial_flowchart() -> Flowchart:
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[Connection(source_id="s", target_id="e")],
    )


def _child_flowchart() -> Flowchart:
    """A trivial start -> end flowchart for the spawned command to run."""
    return Flowchart(
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[Connection(source_id="cs", target_id="ce")],
    )


def _child_flowchart_prompt() -> Flowchart:
    """A child that runs a prompt, so its wall-clock duration is a real >0 ms."""
    return Flowchart(
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "cp": PromptBlock(id="cp", name="ChildAsk", prompt="hi"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[
            Connection(source_id="cs", target_id="cp"),
            Connection(source_id="cp", target_id="ce"),
        ],
    )


def _child_flowchart_exit(code: int) -> Flowchart:
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


class TestExecutionResultTokens:
    async def test_token_fields_populated_from_session_usage(self):
        """``GraphWalker.run`` copies the session's token_usage into the result."""
        session = MockSession()
        session._token_usage = TokenUsage(7, 8, 9, 10)
        walker = GraphWalker(_trivial_flowchart(), session, {}, MockProtocol())

        result = await walker.run()

        assert result.status == "completed"
        assert result.input_tokens == 7
        assert result.output_tokens == 8
        assert result.cache_creation_tokens == 9
        assert result.cache_read_tokens == 10

    async def test_token_fields_default_to_zero_when_session_untracked(self):
        """A run on a zero-usage session reports zeros, not stale values."""
        walker = GraphWalker(_trivial_flowchart(), MockSession(), {}, MockProtocol())

        result = await walker.run()

        assert (
            result.input_tokens,
            result.output_tokens,
            result.cache_creation_tokens,
            result.cache_read_tokens,
        ) == (0, 0, 0, 0)


class TestTokenVariables:
    async def test_all_four_token_variables_receive_child_usage(self):
        """Each token variable holds the joined child's matching component."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            input_tokens_variable="in_1",
            output_tokens_variable="out_1",
            cache_creation_tokens_variable="cc_1",
            cache_read_tokens_variable="cr_1",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _MetricSession(), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        # KeyError here (pre-fix) proves the variable exists only because
        # _exec_wait wrote it; the values prove they are the child's real usage.
        assert result.variables["in_1"] == _CHILD_TOKENS.input_tokens
        assert result.variables["out_1"] == _CHILD_TOKENS.output_tokens
        assert result.variables["cc_1"] == _CHILD_TOKENS.cache_creation_tokens
        assert result.variables["cr_1"] == _CHILD_TOKENS.cache_read_tokens

    async def test_token_variables_absent_leaves_no_spurious_variable(self):
        """No token variables -> nothing is written for tokens."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _MetricSession(), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        distinctive = {101, 202, 303, 404}
        assert not any(
            v in distinctive for v in result.variables.values()
        ), result.variables

    async def test_token_variables_with_join_all_wait(self):
        """An empty (join-all) wait still surfaces the child's token variables."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            input_tokens_variable="in_1",
            cache_read_tokens_variable="cr_1",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=[])  # join all pending
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _MetricSession(), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        assert result.variables["in_1"] == _CHILD_TOKENS.input_tokens
        assert result.variables["cr_1"] == _CHILD_TOKENS.cache_read_tokens


class TestDurationVariable:
    async def test_duration_variable_receives_child_runtime(self):
        """``duration_variable`` holds the joined child's wall-clock duration_ms."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            duration_variable="dur_1",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        # A child that spends measurable time, so duration_ms is a real >0 value.
        walker = GraphWalker(fc, _MetricSession(delay_seconds=0.02), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart_prompt()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        child_ms = walker._spawned_results["cell-1"].duration_ms
        # Exact propagation of the child's own measured runtime...
        assert result.variables["dur_1"] == child_ms
        # ...and it's a real elapsed value, not a stale 0.
        assert result.variables["dur_1"] > 0

    async def test_duration_variable_absent_leaves_no_spurious_variable(self):
        """No ``duration_variable`` -> the child's duration is written nowhere."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _MetricSession(delay_seconds=0.02), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart_prompt()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        child_ms = walker._spawned_results["cell-1"].duration_ms
        assert child_ms > 0  # the child really did run for a measurable time
        assert not any(
            v == child_ms for v in result.variables.values()
        ), result.variables


class TestAllMetricsIndependent:
    async def test_exit_cost_duration_tokens_surface_together(self):
        """All seven per-child metrics are written from one join, independently."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            exit_code_variable="rc_1",
            cost_variable="cost_1",
            duration_variable="dur_1",
            input_tokens_variable="in_1",
            output_tokens_variable="out_1",
            cache_creation_tokens_variable="cc_1",
            cache_read_tokens_variable="cr_1",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        walker = GraphWalker(fc, _MetricSession(), {}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart_exit(3)
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        assert result.variables["rc_1"] == 3
        assert result.variables["cost_1"] == _CHILD_COST
        assert result.variables["in_1"] == _CHILD_TOKENS.input_tokens
        assert result.variables["out_1"] == _CHILD_TOKENS.output_tokens
        assert result.variables["cc_1"] == _CHILD_TOKENS.cache_creation_tokens
        assert result.variables["cr_1"] == _CHILD_TOKENS.cache_read_tokens
        assert isinstance(result.variables["dur_1"], int)
        assert result.variables["dur_1"] >= 0


class TestTemplatedVariableNames:
    async def test_metric_variable_names_are_template_evaluated(self):
        """A spawn's metric variable names resolve {{...}} against variables.

        This is what lets a fan-out give each child a unique target (e.g.
        ``cost_variable: "cost_{{i}}"``) so distinct children never collide on
        one variable -- making assignment (not summation) unambiguously right.
        """
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="cell-1",
            command_name="child-cmd",
            cost_variable="cost_{{i}}",
            duration_variable="dur_{{i}}",
            input_tokens_variable="in_{{i}}",
            cache_read_tokens_variable="cr_{{i}}",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
        fc = _spawn_wait_flowchart(spawn, wait)
        # {{i}} is 7 in the parent's variables at join time.
        walker = GraphWalker(
            fc, _MetricSession(delay_seconds=0.02), {"i": 7}, MockProtocol()
        )

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart_prompt()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "completed", result
        # Names resolved via {{i}} -> the child's metrics landed on the
        # per-index keys...
        assert result.variables["cost_{{i}}".replace("{{i}}", "7")] == _CHILD_COST
        assert result.variables["in_7"] == _CHILD_TOKENS.input_tokens
        assert result.variables["cr_7"] == _CHILD_TOKENS.cache_read_tokens
        assert result.variables["dur_7"] > 0
        # ...and NOT on the un-substituted literal template.
        assert "cost_{{i}}" not in result.variables
        assert "in_{{i}}" not in result.variables
