"""The final-answer query that gives `--json-schema` somewhere to apply.

A flowchart is many turns; a schema names the shape of ONE answer. Applying
it at the CLI level constrained every turn and overrode each block's own
output_schema, which silently broke control flow. Applying it once, after the
flowchart has finished, leaves every block untouched and asks the model --
which still holds the whole conversation -- to restate its final answer in
the requested shape.
"""

import pytest
from flowcoder_engine.__main__ import query_structured_answer


class FakeResult:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text


class FakeSession:
    """Records what it was asked, so the test can assert on the real prompt."""

    def __init__(self, response: str = "", raises: Exception | None = None) -> None:
        self.response = response
        self.raises = raises
        self.prompts: list[str] = []

    async def query(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        if self.raises is not None:
            raise self.raises
        return FakeResult(self.response)


SCHEMA = '{"type":"object","properties":{"digits":{"type":"string"}}}'


async def test_no_schema_issues_no_query():
    """The overwhelming majority of runs. A schema-less flowchart must not
    pay for an extra turn."""
    session = FakeSession(response="irrelevant")
    assert await query_structured_answer(session, None) is None
    assert session.prompts == []


async def test_the_answer_is_parsed_from_the_response():
    session = FakeSession(response='{"digits": "8.185352", "count": 7}')
    assert await query_structured_answer(session, SCHEMA) == {
        "digits": "8.185352", "count": 7,
    }


async def test_the_schema_is_shown_to_the_model():
    session = FakeSession(response="{}")
    await query_structured_answer(session, SCHEMA)
    assert len(session.prompts) == 1
    assert SCHEMA in session.prompts[0]


def test_the_prompt_asks_to_restate_rather_than_recompute():
    """Recomputing would double the cost of the task and risk a DIFFERENT
    answer from the one the flowchart actually produced -- which is the one
    being measured."""
    import asyncio

    session = FakeSession(response="{}")
    asyncio.run(query_structured_answer(session, SCHEMA))
    prompt = session.prompts[0].lower()
    assert "do not" in prompt or "without" in prompt
    assert "recompute" in prompt or "redo" in prompt or "again" in prompt


async def test_fenced_json_is_accepted():
    """Models fence JSON far more often than not."""
    session = FakeSession(response='Here it is:\n```json\n{"digits": "8"}\n```')
    assert await query_structured_answer(session, SCHEMA) == {"digits": "8"}


async def test_unparseable_response_is_none_not_an_exception():
    session = FakeSession(response="I could not comply.")
    assert await query_structured_answer(session, SCHEMA) is None


async def test_a_failing_query_does_not_break_the_run():
    """This runs AFTER the flowchart succeeded. Losing the structured answer
    is a missing field; losing the flowchart's result because the extra turn
    failed would be destroying completed work."""
    session = FakeSession(raises=RuntimeError("session died"))
    assert await query_structured_answer(session, SCHEMA) is None


async def test_a_timeout_does_not_break_the_run():
    session = FakeSession(raises=TimeoutError())
    assert await query_structured_answer(session, SCHEMA) is None


async def test_a_non_object_answer_is_rejected():
    """The host subscripts this; a bare list or string is not an answer."""
    for text in ('["a", "b"]', '"just a string"', "42"):
        session = FakeSession(response=text)
        assert await query_structured_answer(session, SCHEMA) is None, text
