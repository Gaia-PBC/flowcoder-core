"""Tests for the GraphWalker with mock sessions."""

import pytest
from flowcoder_engine.walker import ExecutionError, GraphWalker
from flowcoder_flowchart import (
    BashBlock,
    Connection,
    EndBlock,
    Flowchart,
    PromptBlock,
    StartBlock,
    VariableBlock,
    VariableType,
)

from tests.conftest import MockProtocol, MockSession


@pytest.fixture
def mock_session():
    return MockSession()


@pytest.fixture
def mock_protocol():
    return MockProtocol()


class TestSimpleFlow:
    async def test_start_prompt_end(self, simple_flowchart, mock_session, mock_protocol):
        walker = GraphWalker(simple_flowchart, mock_session, {"$1": "World"}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"
        assert len(result.log) == 3  # start, prompt, end

    async def test_prompt_receives_resolved_template(self, simple_flowchart, mock_protocol):
        session = MockSession()
        walker = GraphWalker(simple_flowchart, session, {"$1": "Alice"}, mock_protocol)
        await walker.run()

        # The mock session should have been queried
        assert session._call_count == 1

    async def test_empty_flowchart_no_start(self, mock_session, mock_protocol):
        fc = Flowchart(blocks={"e": EndBlock(id="e")})
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"
        assert len(result.log) == 0


class TestBranching:
    async def test_branch_true(self, branch_flowchart, mock_session, mock_protocol):
        walker = GraphWalker(
            branch_flowchart, mock_session, {"flag": True}, mock_protocol
        )
        result = await walker.run()
        assert result.status == "completed"
        # Should have hit: start -> branch -> ok
        block_names = [e.block_name for e in result.log]
        assert "OK" in block_names
        assert "Fail" not in block_names

    async def test_branch_false(self, branch_flowchart, mock_session, mock_protocol):
        walker = GraphWalker(
            branch_flowchart, mock_session, {"flag": False}, mock_protocol
        )
        result = await walker.run()
        block_names = [e.block_name for e in result.log]
        assert "Fail" in block_names
        assert "OK" not in block_names

    async def test_branch_missing_var_is_falsy(
        self, branch_flowchart, mock_session, mock_protocol
    ):
        walker = GraphWalker(branch_flowchart, mock_session, {}, mock_protocol)
        result = await walker.run()
        block_names = [e.block_name for e in result.log]
        assert "Fail" in block_names

    async def test_branch_string_false(
        self, branch_flowchart, mock_session, mock_protocol
    ):
        walker = GraphWalker(
            branch_flowchart, mock_session, {"flag": "false"}, mock_protocol
        )
        result = await walker.run()
        block_names = [e.block_name for e in result.log]
        assert "Fail" in block_names

    async def test_branch_string_true(
        self, branch_flowchart, mock_session, mock_protocol
    ):
        walker = GraphWalker(
            branch_flowchart, mock_session, {"flag": "yes"}, mock_protocol
        )
        result = await walker.run()
        block_names = [e.block_name for e in result.log]
        assert "OK" in block_names


class TestVariables:
    async def test_variable_set(self, variable_flowchart, mock_session, mock_protocol):
        walker = GraphWalker(variable_flowchart, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.variables["name"] == "World"

    async def test_variable_number(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "v": VariableBlock(
                    id="v", variable_name="x", variable_value="42",
                    variable_type=VariableType.NUMBER,
                ),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="v"),
                Connection(source_id="v", target_id="e"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.variables["x"] == 42.0

    async def test_variable_boolean(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "v": VariableBlock(
                    id="v", variable_name="flag", variable_value="true",
                    variable_type=VariableType.BOOLEAN,
                ),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="v"),
                Connection(source_id="v", target_id="e"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.variables["flag"] is True

    async def test_variable_json(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "v": VariableBlock(
                    id="v", variable_name="data",
                    variable_value='{"key": "value"}',
                    variable_type=VariableType.JSON,
                ),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="v"),
                Connection(source_id="v", target_id="e"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.variables["data"] == {"key": "value"}

    async def test_variable_template_resolution(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "v1": VariableBlock(
                    id="v1", variable_name="greeting", variable_value="Hello",
                ),
                "v2": VariableBlock(
                    id="v2", variable_name="msg",
                    variable_value="{{greeting}}, World!",
                ),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="v1"),
                Connection(source_id="v1", target_id="v2"),
                Connection(source_id="v2", target_id="e"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.variables["msg"] == "Hello, World!"


class TestBash:
    async def test_bash_capture_output(self, bash_flowchart, mock_session, mock_protocol):
        walker = GraphWalker(bash_flowchart, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"
        assert result.variables.get("result") == "hello"

    async def test_bash_failure_halts(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "b": BashBlock(id="b", command="exit 1", continue_on_error=False),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="b"),
                Connection(source_id="b", target_id="e"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "halted"

    async def test_bash_continue_on_error(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "b": BashBlock(
                    id="b", command="exit 1",
                    continue_on_error=True,
                    exit_code_variable="rc",
                ),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="b"),
                Connection(source_id="b", target_id="e"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"
        assert result.variables["rc"] == 1

    async def test_bash_template_in_command(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "v": VariableBlock(id="v", variable_name="msg", variable_value="hi"),
                "b": BashBlock(
                    id="b", command="echo {{msg}}",
                    capture_output=True, output_variable="out",
                ),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="v"),
                Connection(source_id="v", target_id="b"),
                Connection(source_id="b", target_id="e"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.variables["out"] == "hi"


class TestRefresh:
    async def test_refresh_clears_session(
        self, refresh_flowchart, mock_protocol
    ):
        session = MockSession(["First response", "Second response"])
        walker = GraphWalker(refresh_flowchart, session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"
        assert session._clear_count == 1


class TestMultiSession:
    async def test_multi_session_uses_single_session(
        self, multi_session_flowchart, mock_protocol
    ):
        """Multi-session flowcharts use the single main session (with warning)."""
        session = MockSession(["Deployed!", "Looks good!"])
        walker = GraphWalker(
            multi_session_flowchart, session, {"$1": "main"}, mock_protocol
        )
        result = await walker.run()
        assert result.status == "completed"
        assert session._call_count == 2  # both blocks used same session
        # Should have logged warnings about non-default sessions
        assert any("Multi-session not yet supported" in log for log in mock_protocol.logs)


class TestLooping:
    async def test_loop_completes(self, loop_flowchart, mock_protocol):
        session = MockSession()
        walker = GraphWalker(loop_flowchart, session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"
        assert result.variables["done"] is True

    async def test_safety_limit(self, mock_protocol):
        """Infinite loop should hit safety limit."""
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "v": VariableBlock(id="v", variable_name="x", variable_value="1"),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="v"),
                Connection(source_id="v", target_id="v"),  # loop back to self
            ],
        )
        session = MockSession()
        walker = GraphWalker(fc, session, {}, mock_protocol, max_blocks=10)
        with pytest.raises(ExecutionError, match="Safety limit"):
            await walker.run()


class TestProtocolMessages:
    async def test_emits_block_start_and_complete(
        self, simple_flowchart, mock_session, mock_protocol
    ):
        walker = GraphWalker(simple_flowchart, mock_session, {"$1": "X"}, mock_protocol)
        await walker.run()

        block_starts = [
            m for m in mock_protocol.messages
            if m.get("subtype") == "block_start"
        ]
        block_completes = [
            m for m in mock_protocol.messages
            if m.get("subtype") == "block_complete"
        ]

        assert len(block_starts) == 3  # start, prompt, end
        assert len(block_completes) == 3
        assert all(m["data"]["success"] for m in block_completes)

    async def test_logs_execution(
        self, simple_flowchart, mock_session, mock_protocol
    ):
        walker = GraphWalker(simple_flowchart, mock_session, {"$1": "X"}, mock_protocol)
        await walker.run()
        assert any("Executing block" in log for log in mock_protocol.logs)


def _timeout_flowchart(timeout_seconds: int) -> Flowchart:
    """start -> prompt(timeout_seconds=…) -> end"""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "p": PromptBlock(
                id="p",
                name="Slow",
                prompt="Anything",
                timeout_seconds=timeout_seconds,
            ),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="p"),
            Connection(source_id="p", target_id="e"),
        ],
    )


class TestBlockTimeout:
    def test_default_timeout_value(self):
        """PromptBlock defaults to 6h (21600s) when not specified."""
        block = PromptBlock(id="p", name="x", prompt="hi")
        assert block.timeout_seconds == 21600

    def test_timeout_from_json(self):
        """An override on PromptBlock round-trips through Pydantic."""
        block = PromptBlock.model_validate({
            "id": "p",
            "name": "x",
            "prompt": "hi",
            "timeout_seconds": 30,
        })
        assert block.timeout_seconds == 30
        # And a dump preserves it
        assert block.model_dump()["timeout_seconds"] == 30

    async def test_timeout_fires_with_hanging_session(self, mock_protocol):
        session = MockSession(delay_seconds=60)
        fc = _timeout_flowchart(timeout_seconds=1)
        walker = GraphWalker(fc, session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "halted"
        # Session.stop() should have been called to kill the hung subprocess
        assert session._stop_count >= 1

    async def test_block_timeout_event_emitted(self, mock_protocol):
        session = MockSession(delay_seconds=60)
        fc = _timeout_flowchart(timeout_seconds=1)
        walker = GraphWalker(fc, session, {}, mock_protocol)
        await walker.run()

        timeouts = [
            m for m in mock_protocol.messages
            if m.get("subtype") == "block_timeout"
        ]
        assert len(timeouts) == 1
        data = timeouts[0]["data"]
        assert data["block_id"] == "p"
        assert data["block_name"] == "Slow"
        assert data["block_type"] == "prompt"
        assert isinstance(data["elapsed_ms"], int)
        assert data["elapsed_ms"] >= 0
        assert isinstance(data["timeout_seconds"], int)
        assert data["timeout_seconds"] == 1

    async def test_block_complete_carries_session_id(self, mock_protocol):
        """After a prompt block, subsequent block_completes carry the session_id."""
        session = MockSession(session_id="sess-abc-123")
        fc = _timeout_flowchart(timeout_seconds=10)
        walker = GraphWalker(fc, session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"

        block_completes = [
            m for m in mock_protocol.messages
            if m.get("subtype") == "block_complete"
        ]
        # MockSession exposes its session_id from the start (mock), so every
        # block_complete carries the id field.
        assert len(block_completes) == 3  # start, prompt, end
        for msg in block_completes:
            assert msg["data"].get("session_id") == "sess-abc-123"

    async def test_no_timeout_under_limit(self, mock_protocol):
        """A fast prompt under its limit completes normally."""
        session = MockSession(delay_seconds=0)
        fc = _timeout_flowchart(timeout_seconds=10)
        walker = GraphWalker(fc, session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"
        # No timeout events fired
        assert not any(
            m.get("subtype") == "block_timeout"
            for m in mock_protocol.messages
        )
        # session.stop() was not called by the walker (no timeout)
        assert session._stop_count == 0

    async def test_block_complete_session_id_omitted_when_none(self, mock_protocol):
        """If the session has no id yet, block_complete should not carry session_id."""
        session = MockSession(session_id=None)
        fc = _timeout_flowchart(timeout_seconds=10)
        walker = GraphWalker(fc, session, {}, mock_protocol)
        await walker.run()
        block_completes = [
            m for m in mock_protocol.messages
            if m.get("subtype") == "block_complete"
        ]
        for msg in block_completes:
            assert "session_id" not in msg["data"]
