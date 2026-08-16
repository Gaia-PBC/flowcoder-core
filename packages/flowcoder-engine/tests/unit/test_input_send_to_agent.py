"""An input block can capture user text without sending it to the agent.

``input_variable`` gave a flowchart the user's raw text, but the block still
queried the agent with it unconditionally -- so "ask the user something and
branch on it" cost an agent turn nobody wanted, and the agent saw input meant
for the flowchart alone.  ``send_to_agent=False`` captures the text and returns
without querying; a later ``prompt`` block can send it if and when it should be
sent, via ``{{user_text}}``.

Default is ``True``, so every existing input block behaves exactly as before.
"""

from __future__ import annotations

import asyncio

from flowcoder_engine.subprocess import ReadTimeoutError
from flowcoder_engine.walker import GraphWalker
from flowcoder_flowchart import (
    Connection,
    EndBlock,
    Flowchart,
    InputBlock,
    StartBlock,
)

from tests.conftest import MockProtocol, MockSession


def _input_flowchart(block: InputBlock) -> Flowchart:
    return Flowchart(
        blocks={
            "s": StartBlock(id="s"),
            "i": block,
            "e": EndBlock(id="e"),
        },
        connections=[
            Connection(source_id="s", target_id="i"),
            Connection(source_id="i", target_id="e"),
        ],
    )


class _InboxProtocol(MockProtocol):
    """Delivers one ``input_response`` for block ``i``, as the runner would."""

    def __init__(self, content: str = "user says hi"):
        super().__init__()
        self._inbox: asyncio.Queue[dict] = asyncio.Queue()
        self._inbox.put_nowait(
            {"type": "input_response", "block_id": "i", "content": content}
        )

    async def read_message(self):
        return await self._inbox.get()

    def push_message(self, msg):
        self._inbox.put_nowait(msg)


class TestSendToAgentField:
    def test_defaults_to_true(self):
        assert InputBlock(id="i", name="Ask").send_to_agent is True

    def test_round_trips_through_json(self):
        block = InputBlock.model_validate(
            {"id": "i", "name": "Ask", "send_to_agent": False}
        )
        assert block.send_to_agent is False
        assert block.model_dump()["send_to_agent"] is False


class TestDefaultStillSends:
    async def test_default_queries_the_agent_exactly_once(self):
        # The invariant: unchanged behaviour for every block that does not
        # opt out.
        session = MockSession()
        walker = GraphWalker(
            _input_flowchart(InputBlock(id="i", name="Ask", output_variable="reply")),
            session,
            {},
            _InboxProtocol(),
        )

        result = await walker.run()

        assert result.status == "completed", result
        assert session._call_count == 1
        assert result.variables.get("reply") == "Mock response"


class TestSendToAgentFalse:
    async def test_does_not_query_the_agent(self):
        session = MockSession()
        walker = GraphWalker(
            _input_flowchart(InputBlock(id="i", name="Ask", send_to_agent=False)),
            session,
            {},
            _InboxProtocol(),
        )

        result = await walker.run()

        assert result.status == "completed", result
        # The point of the flag: the agent is never asked.
        assert session._call_count == 0

    async def test_still_captures_the_input_variable(self):
        session = MockSession()
        walker = GraphWalker(
            _input_flowchart(
                InputBlock(
                    id="i", name="Ask", input_variable="user_text", send_to_agent=False
                )
            ),
            session,
            {},
            _InboxProtocol("blue"),
        )

        result = await walker.run()

        assert result.status == "completed", result
        assert result.variables.get("user_text") == "blue"
        assert session._call_count == 0

    async def test_leaves_output_variable_unset(self):
        # There is no agent reply to store, so output_variable stays absent
        # rather than being filled with something misleading.
        walker = GraphWalker(
            _input_flowchart(
                InputBlock(
                    id="i", name="Ask", output_variable="reply", send_to_agent=False
                )
            ),
            MockSession(),
            {},
            _InboxProtocol(),
        )

        result = await walker.run()

        assert result.status == "completed", result
        assert "reply" not in result.variables


class TestCaptureHappensBeforeSend:
    async def test_input_variable_survives_a_failing_agent_call(self):
        # The capture is documented as happening "before sending it to the
        # agent", but was written after the query -- so an agent that died
        # took the user's text with it, even though the block had already
        # received it.
        class TimingOutSession(MockSession):
            async def query(self, *args, **kwargs):
                raise ReadTimeoutError("CLI idle during query")

        walker = GraphWalker(
            _input_flowchart(
                InputBlock(id="i", name="Ask", input_variable="user_text")
            ),
            TimingOutSession(),
            {},
            _InboxProtocol("do not lose me"),
        )

        result = await walker.run()

        assert result.status != "completed"
        assert result.variables.get("user_text") == "do not lose me"
