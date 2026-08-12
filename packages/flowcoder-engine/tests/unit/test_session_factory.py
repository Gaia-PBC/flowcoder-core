"""Tests for SessionFactory and cross-backend spawning."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from flowcoder_engine.session import BaseSession
from flowcoder_engine.session_factory import SessionFactory


class StubSession(BaseSession):
    """Minimal session for testing."""

    def __init__(self, name: str, backend: str, model: str | None = None):
        self._name = name
        self._backend = backend
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    @property
    def session_id(self) -> str | None:
        return None

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def is_running(self) -> bool:
        return False

    def clone(self, name: str, cwd: str | None = None) -> StubSession:
        return StubSession(name, self._backend, self._model)

    def with_model(self, model: str) -> StubSession:
        return StubSession(self._name, self._backend, model)

    async def start(self) -> None:
        pass

    async def query(self, prompt: str, block_id: str = "", block_name: str = ""):
        from flowcoder_engine.session import QueryResult
        return QueryResult(response_text=f"{self._backend}:{prompt}")

    async def clear(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class TestSessionFactory:
    def test_register_and_create(self):
        factory = SessionFactory()
        factory.register("test", lambda name, model: StubSession(name, "test", model))
        session = factory.create("test", "agent-1")
        assert isinstance(session, StubSession)
        assert session.name == "agent-1"
        assert session._backend == "test"

    def test_create_with_model(self):
        factory = SessionFactory()
        factory.register("test", lambda name, model: StubSession(name, "test", model))
        session = factory.create("test", "agent-1", model="gpt-5")
        assert session._model == "gpt-5"

    def test_unknown_backend_raises(self):
        factory = SessionFactory()
        with pytest.raises(ValueError, match="Unknown backend 'missing'"):
            factory.create("missing", "agent-1")

    def test_backends_property(self):
        factory = SessionFactory()
        factory.register("claude", lambda n, m: StubSession(n, "claude", m))
        factory.register("codex", lambda n, m: StubSession(n, "codex", m))
        assert sorted(factory.backends) == ["claude", "codex"]

    def test_multiple_backends(self):
        factory = SessionFactory()
        factory.register("claude", lambda n, m: StubSession(n, "claude", m))
        factory.register("codex", lambda n, m: StubSession(n, "codex", m))

        s1 = factory.create("claude", "agent-claude")
        s2 = factory.create("codex", "agent-codex")

        assert s1._backend == "claude"
        assert s2._backend == "codex"


class TestCrossBackendSpawn:
    """Test that walker uses factory for cross-backend spawning."""

    @pytest.mark.asyncio
    async def test_spawn_with_backend_uses_factory(self):
        from flowcoder_flowchart import BlockType, Flowchart, SpawnBlock, WaitBlock

        from flowcoder_engine.walker import GraphWalker

        # Create a minimal flowchart for the spawned command
        child_flowchart = Flowchart(
            blocks={
                "start": {"id": "start", "type": "start", "name": "S", "position": {"x": 0, "y": 0}},
                "end": {"id": "end", "type": "end", "name": "E", "position": {"x": 0, "y": 100}},
            },
            connections=[{"id": "c1", "source_block_id": "start", "target_block_id": "end", "is_true_path": True}],
        )

        # Mock resolve_command to return a command with child_flowchart
        from unittest.mock import patch, MagicMock

        mock_cmd = MagicMock()
        mock_cmd.flowchart = child_flowchart

        # Main flowchart: spawn -> wait -> end
        main_flowchart = Flowchart(
            blocks={
                "start": {"id": "start", "type": "start", "name": "S", "position": {"x": 0, "y": 0}},
                "spawn": {
                    "id": "spawn",
                    "type": "spawn",
                    "name": "SPAWN",
                    "position": {"x": 0, "y": 100},
                    "agent_name": "worker",
                    "command_name": "test-cmd",
                    "backend": "codex",
                },
                "wait": {
                    "id": "wait",
                    "type": "wait",
                    "name": "WAIT",
                    "position": {"x": 0, "y": 200},
                    "wait_for": ["worker"],
                },
                "end": {"id": "end", "type": "end", "name": "E", "position": {"x": 0, "y": 300}},
            },
            connections=[
                {"id": "c1", "source_block_id": "start", "target_block_id": "spawn", "is_true_path": True},
                {"id": "c2", "source_block_id": "spawn", "target_block_id": "wait", "is_true_path": True},
                {"id": "c3", "source_block_id": "wait", "target_block_id": "end", "is_true_path": True},
            ],
        )

        # Main session is "claude" backend
        main_session = StubSession("main", "claude")

        # Factory that creates "codex" sessions
        factory = SessionFactory()
        created_sessions = []

        def create_codex(name, model):
            s = StubSession(name, "codex", model)
            created_sessions.append(s)
            return s

        factory.register("codex", create_codex)

        protocol = MagicMock()
        protocol.log = MagicMock()

        walker = GraphWalker(
            main_flowchart,
            main_session,
            {},
            protocol,
            search_paths=[],
            session_factory=factory,
        )

        with patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd):
            result = await walker.run()

        assert result.status == "completed"
        assert len(created_sessions) == 1
        assert created_sessions[0]._backend == "codex"
        assert created_sessions[0].name == "worker"
