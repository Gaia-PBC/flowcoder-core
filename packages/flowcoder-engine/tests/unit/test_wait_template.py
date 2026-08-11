"""Regression tests: a WaitBlock's ``wait_for`` entries are template-evaluated.

A spawn block's ``agent_name`` is run through ``evaluate_template`` before the
child task is stored (so ``spawn "w-{{i}}"`` registers under the resolved name
``"w-1"``), but ``wait_for`` entries used to be looked up *literally*. That
meant ``wait_for: ["w-{{i}}"]`` searched ``_spawned_tasks`` for the raw string
``"w-{{i}}"``, never matched the ``"w-1"`` key, and the wait failed with
``No spawned agent named 'w-{{i}}'`` -- defeating the very static-naming the
validator forces authors into for looped/templated fan-out.

These tests pin that ``wait_for`` is resolved against the walker variables
*before* the pending-task lookup, mirroring ``_exec_spawn``'s handling of
``agent_name``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def _child_flowchart() -> Flowchart:
    """A trivial start -> end flowchart for the spawned command to run."""
    return Flowchart(
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[Connection(source_id="cs", target_id="ce")],
    )


class TestWaitForTemplateResolved:
    async def test_templated_wait_for_matches_templated_spawn(self):
        """``wait_for: ["w-{{i}}"]`` must join the child spawned as ``"w-{{i}}"``.

        Both names resolve to ``"w-1"`` against ``{"i": "1"}``; before the fix
        the literal lookup missed and the run halted.
        """
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="w-{{i}}",
            command_name="child-cmd",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["w-{{i}}"])
        fc = Flowchart(
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
        protocol = MockProtocol()
        walker = GraphWalker(fc, MockSession(), {"i": "1"}, protocol)

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        # Unfixed: halts with "No spawned agent named 'w-{{i}}'".
        assert result.status == "completed", result
        assert not any(
            "{{i}}" in log for log in protocol.logs
        ), "wait_for should have been resolved before lookup"

    async def test_templated_wait_for_reports_the_resolved_name_on_miss(self):
        """A genuine miss reports the *resolved* name, not the raw template.

        If the wait names an agent that was never spawned, the error must
        reference what was actually looked up (``"x-1"``), proving the entry
        was templated before the lookup rather than passed through verbatim.
        """
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="w-{{i}}",
            command_name="child-cmd",
        )
        wait = WaitBlock(id="wait", name="Wait", wait_for=["x-{{i}}"])
        fc = Flowchart(
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
        walker = GraphWalker(fc, MockSession(), {"i": "1"}, MockProtocol())

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ):
            result = await walker.run()

        assert result.status == "halted"
        errors = [e.result.error for e in result.log if e.result.error]
        assert any("x-1" in err for err in errors), errors
        assert not any("{{i}}" in err for err in errors), errors
