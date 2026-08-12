"""ClaudeSession accumulates per-turn token usage by SUMMING across queries.

The Claude CLI's ``result.usage`` block is reported *per turn* (verified
against real CLI output), unlike the cumulative ``total_cost_usd``.  These
tests pin that ClaudeSession sums each query's usage into ``token_usage`` — a
regression to the cost-style cumulative-delta would compute the second query's
contribution wrong (negative deltas from a per-turn value).

Driven with a fake process so the parsing/accumulation path runs for real
without a subprocess or any API call.
"""

from __future__ import annotations

from typing import Any

import pytest
from flowcoder_engine.session import ClaudeSession, TokenUsage


class _FakeProcess:
    """Feeds scripted stream-json messages to ``ClaudeSession.query``.

    ``write`` of a ``user`` message advances to the next scripted turn; ``read``
    pops that turn's messages then returns ``None`` (mirrors the real subprocess
    yielding None at end of a turn's output).
    """

    def __init__(self, turns: list[list[dict[str, Any]]]):
        self._turns = turns
        self._turn = -1
        self._queue: list[dict[str, Any]] = []
        self.is_running = True

    async def write(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == "user":
            self._turn += 1
            self._queue = list(self._turns[self._turn])

    async def read(self, timeout: float | None = None) -> dict[str, Any] | None:
        if self._queue:
            return self._queue.pop(0)
        return None


def _result(usage: dict[str, int], cost: float) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "result": "ok",
        "duration_ms": 1,
        "total_cost_usd": cost,
        "usage": usage,
        "session_id": "fake",
    }


def _session_with(turns: list[list[dict[str, Any]]]) -> ClaudeSession:
    s = ClaudeSession(name="t", claude_cmd=["claude"])
    s._process = _FakeProcess(turns)  # type: ignore[assignment]
    return s


class TestTokenAccounting:
    async def test_single_query_populates_token_usage(self):
        """One result's usage lands on both the QueryResult and the session."""
        s = _session_with([[
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            },
            _result(
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 100,
                },
                0.1,
            ),
        ]])

        r = await s.query("hello")

        assert r.token_usage == TokenUsage(10, 5, 2, 100)
        assert s.token_usage == TokenUsage(10, 5, 2, 100)

    async def test_usage_sums_across_queries_not_delta(self):
        """Two queries' usage is SUMMED, proving we don't reuse cost's delta."""
        s = _session_with([
            [_result(
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 100,
                },
                0.1,
            )],
            [_result(
                {
                    "input_tokens": 3,
                    "output_tokens": 7,
                    "cache_creation_input_tokens": 1,
                    "cache_read_input_tokens": 200,
                },
                0.25,
            )],
        ])

        await s.query("q1")
        await s.query("q2")

        # SUM (10+3, 5+7, 2+1, 100+200).  A cost-style delta on the per-turn
        # ``usage`` would yield garbage here (e.g. input 3-10 = -7).
        assert s.token_usage == TokenUsage(13, 12, 3, 300)
        # Cost keeps its own cumulative-delta semantics: last cumulative wins.
        assert s.total_cost == pytest.approx(0.25)

    async def test_missing_usage_block_is_zero_not_error(self):
        """A result with no ``usage`` accumulates zeros rather than raising."""
        s = _session_with([[_result({}, 0.0)]])

        r = await s.query("q")

        assert r.token_usage == TokenUsage(0, 0, 0, 0)
        assert s.token_usage == TokenUsage(0, 0, 0, 0)
