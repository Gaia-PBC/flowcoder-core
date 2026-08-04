"""Tests for walker extensions: exit, halt/resume, input, try/finally cleanup."""

from __future__ import annotations

import asyncio

import pytest
from flowcoder_engine.subprocess import ReadTimeoutError
from flowcoder_engine.walker import BlockResult, ExecutionResult, GraphWalker
from flowcoder_flowchart import (
    Connection,
    EndBlock,
    ExitBlock,
    Flowchart,
    InputBlock,
    PromptBlock,
    StartBlock,
    VariableBlock,
)

from tests.conftest import MockProtocol, MockSession


@pytest.fixture
def mock_session():
    return MockSession()


@pytest.fixture
def mock_protocol():
    return MockProtocol()


class TestExitBlock:
    async def test_exit_code_zero(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "x": ExitBlock(id="x", name="Done", exit_code=0, exit_message="success"),
            },
            connections=[Connection(source_id="s", target_id="x")],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "completed"
        assert result.exit_code == 0

    async def test_exit_code_nonzero(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "x": ExitBlock(id="x", name="Fail", exit_code=1, exit_message="error"),
            },
            connections=[Connection(source_id="s", target_id="x")],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.status == "exited"
        assert result.exit_code == 1

    async def test_exit_message_template(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "v": VariableBlock(
                    id="v", variable_name="reason", variable_value="timeout"
                ),
                "x": ExitBlock(
                    id="x", name="Bail", exit_code=2, exit_message="Failed: {{reason}}"
                ),
            },
            connections=[
                Connection(source_id="s", target_id="v"),
                Connection(source_id="v", target_id="x"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        result = await walker.run()
        assert result.exit_code == 2
        log_entry = [e for e in result.log if e.block_type == "exit"][0]
        assert "timeout" in log_entry.result.output


class TestHaltResume:
    async def test_halt_stops_execution(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "p": PromptBlock(id="p", prompt="hello"),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="p"),
                Connection(source_id="p", target_id="e"),
            ],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol)
        walker.halt()
        result = await walker.run()
        assert result.status == "halted"
        # Halt checked before any block executes
        assert len(result.log) == 0

    async def test_resume_clears_halt(self, mock_session, mock_protocol):
        walker = GraphWalker(
            Flowchart(
                blocks={"s": StartBlock(id="s"), "e": EndBlock(id="e")},
                connections=[Connection(source_id="s", target_id="e")],
            ),
            mock_session,
            {},
            mock_protocol,
        )
        walker.halt()
        walker.resume()
        result = await walker.run()
        assert result.status == "completed"


class TestBlockResult:
    def test_exit_classmethod(self):
        r = BlockResult.exit(code=42, message="done")
        assert r.success is True
        assert r.exit_code == 42
        assert r.output == "done"

    def test_ok_has_no_exit_code(self):
        r = BlockResult.ok(output="hello")
        assert r.exit_code is None


class TestInputBlock:
    async def test_input_captures_response(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "i": InputBlock(id="i", name="Ask", output_variable="user_input"),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="i"),
                Connection(source_id="i", target_id="e"),
            ],
        )

        # Pre-load the inbox with an input_response
        mock_protocol.push_message = lambda msg: None  # ignore re-queued msgs
        # We need to make read_message return our response
        response_msg = {"type": "input_response", "block_id": "i", "content": "user says hi"}

        original_read = None

        class InboxProtocol(MockProtocol):
            def __init__(self):
                super().__init__()
                self._inbox: asyncio.Queue[dict] = asyncio.Queue()
                self._inbox.put_nowait(response_msg)

            async def read_message(self):
                return await self._inbox.get()

            def push_message(self, msg):
                self._inbox.put_nowait(msg)

        proto = InboxProtocol()
        walker = GraphWalker(fc, mock_session, {}, proto)
        result = await walker.run()
        assert result.status == "completed"
        assert result.variables.get("user_input") == "Mock response"

    async def test_input_empty_content(self, mock_session, mock_protocol):
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "i": InputBlock(id="i", name="Ask"),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="i"),
                Connection(source_id="i", target_id="e"),
            ],
        )

        class InboxProtocol(MockProtocol):
            def __init__(self):
                super().__init__()
                self._inbox: asyncio.Queue[dict] = asyncio.Queue()
                self._inbox.put_nowait(
                    {"type": "input_response", "block_id": "i", "content": ""}
                )

            async def read_message(self):
                return await self._inbox.get()

            def push_message(self, msg):
                self._inbox.put_nowait(msg)

        proto = InboxProtocol()
        walker = GraphWalker(fc, mock_session, {}, proto)
        result = await walker.run()
        assert result.status == "completed"

    async def test_input_read_timeout_reports_block_timeout(self, mock_protocol):
        """An idle CLI during an input block must fail the block, not raise.

        The ReadTimeoutError handler reports the block's configured timeout,
        so timeout_seconds has to exist on InputBlock.  It was originally
        added to PromptBlock alone, which made this path raise
        AttributeError -- on exactly the scenario the handler exists for.
        """

        class TimingOutSession(MockSession):
            async def query(self, *args, **kwargs):
                raise ReadTimeoutError("CLI idle during query")

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "i": InputBlock(id="i", name="Ask"),
                "e": EndBlock(id="e"),
            },
            connections=[
                Connection(source_id="s", target_id="i"),
                Connection(source_id="i", target_id="e"),
            ],
        )

        class InboxProtocol(MockProtocol):
            def __init__(self):
                super().__init__()
                self._inbox: asyncio.Queue[dict] = asyncio.Queue()
                self._inbox.put_nowait(
                    {"type": "input_response", "block_id": "i", "content": "hello"}
                )

            async def read_message(self):
                return await self._inbox.get()

            def push_message(self, msg):
                self._inbox.put_nowait(msg)

        proto = InboxProtocol()
        walker = GraphWalker(fc, TimingOutSession(), {}, proto)
        result = await walker.run()

        assert result.status != "completed"
        timeouts = [
            m for m in proto.messages if m.get("subtype") == "block_timeout"
        ]
        assert timeouts, "expected a block_timeout event for the idle input block"
        data = timeouts[0]["data"]
        assert data["block_id"] == "i"
        assert data["timeout_seconds"] == 21600


class TestCleanup:
    async def test_cleanup_on_error(self, mock_session, mock_protocol):
        """Verify that try/finally cleanup runs even on errors."""
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "e": EndBlock(id="e"),
            },
            connections=[Connection(source_id="s", target_id="e")],
        )
        walker = GraphWalker(fc, mock_session, {}, mock_protocol, max_blocks=0)
        with pytest.raises(Exception):
            await walker.run()
        # Cleanup should have run (no spawned tasks, but no crash either)
        assert walker._spawned_tasks == {}
        assert walker._spawned_sessions == {}
