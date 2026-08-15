"""Regression tests: a SpawnBlock's ``model`` field is template-evaluated.

Every other spawn field (``agent_name``, ``command_name``, ``arguments``) is
run through ``evaluate_template`` before use, but ``model`` used to be passed
to the child session raw -- so ``model: "{{child_model}}"`` (or ``"$1"``)
reached ``with_model``/the session factory as the literal template string,
which broke selecting the child model from a flowchart argument.

These tests pin that ``model`` is resolved against the walker variables
*before* the child session is created, for both spawn code paths: the
session-factory path and the ``with_model`` path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from flowcoder_engine.session_factory import SessionFactory
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


def _spawn_wait_flowchart(spawn: SpawnBlock) -> Flowchart:
    """start -> <spawn> -> wait(worker) -> end."""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "spawn": spawn,
            "wait": WaitBlock(id="wait", name="Wait", wait_for=["worker"]),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="spawn"),
            Connection(source_id="spawn", target_id="wait"),
            Connection(source_id="wait", target_id="e"),
        ],
    )


class _WithModelRecordingSession(MockSession):
    """MockSession that records the argument passed to ``with_model``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_model_args: list[str] = []

    def with_model(self, model: str) -> MockSession:
        self.with_model_args.append(model)
        return super().with_model(model)


class TestSpawnModelTemplateResolved:
    async def test_with_model_path_resolves_template(self):
        """`with_model` must receive the resolved model, not the raw template."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="worker",
            command_name="child-cmd",
            model="{{child_model}}",
        )
        session = _WithModelRecordingSession()
        protocol = MockProtocol()
        walker = GraphWalker(
            _spawn_wait_flowchart(spawn),
            session,
            {"child_model": "claude-sonnet-4-5"},
            protocol,
        )

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        assert result.status == "completed"
        # The resolved value reaches with_model; the literal template never does.
        assert session.with_model_args == ["claude-sonnet-4-5"]
        assert "{{child_model}}" not in session.with_model_args
        assert any("claude-sonnet-4-5" in log for log in protocol.logs)
        assert not any("{{child_model}}" in log for log in protocol.logs)

    async def test_factory_path_resolves_template(self):
        """The session factory must receive the resolved model, not the template."""
        spawn = SpawnBlock(
            id="spawn",
            name="Spawn",
            agent_name="worker",
            command_name="child-cmd",
            backend="codex",
            model="{{child_model}}",
        )

        created: list[tuple[str, str | None]] = []

        def _create(name, model, env=None):
            created.append((name, model))
            return MockSession()

        factory = SessionFactory()
        factory.register("codex", _create)

        protocol = MockProtocol()
        walker = GraphWalker(
            _spawn_wait_flowchart(spawn),
            MockSession(),
            {"child_model": "gpt-5-codex"},
            protocol,
            session_factory=factory,
        )

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _child_flowchart()
        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        assert result.status == "completed"
        assert len(created) == 1
        # The factory receives the resolved model, not the literal template.
        assert created[0] == ("worker", "gpt-5-codex")
        assert any("gpt-5-codex" in log for log in protocol.logs)
        assert not any("{{child_model}}" in log for log in protocol.logs)
