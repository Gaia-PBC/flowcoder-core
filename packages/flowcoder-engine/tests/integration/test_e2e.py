"""End-to-end tests across both packages.

Validates that example flowcharts/commands load correctly,
pass validation, and can be executed by the walker with mocked sessions.
"""

import sys
from pathlib import Path

# ── flowchart lib tests ──────────────────────────────────────────────────
from flowcoder_flowchart import (
    Command,
    load_command,
    validate,
)

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"


def test_load_simple_example():
    cmd = load_command(EXAMPLES / "simple.json")
    assert cmd.name == "chat"
    assert len(cmd.flowchart.blocks) == 3
    result = validate(cmd.flowchart)
    assert result.valid, f"Validation errors: {result.errors}"


def test_load_multi_session_example():
    cmd = load_command(EXAMPLES / "multi_session.json")
    assert cmd.name == "code-review"
    assert len(cmd.flowchart.sessions) == 2
    assert "coder" in cmd.flowchart.sessions
    assert "reviewer" in cmd.flowchart.sessions
    result = validate(cmd.flowchart)
    assert result.valid, f"Validation errors: {result.errors}"


def test_multi_session_blocks_reference_sessions():
    cmd = load_command(EXAMPLES / "multi_session.json")
    fc = cmd.flowchart
    # All non-start/end blocks should reference a session
    for bid, block in fc.blocks.items():
        if block.type == "prompt":
            assert block.session in ("coder", "reviewer", "default"), (
                f"Block {bid} has unexpected session: {block.session}"
            )


def test_command_roundtrip():
    """Load, dump, reload — should be identical."""
    cmd = load_command(EXAMPLES / "multi_session.json")
    data = cmd.model_dump(mode="json")
    restored = Command.model_validate(data)
    assert restored.name == cmd.name
    assert len(restored.flowchart.blocks) == len(cmd.flowchart.blocks)
    assert len(restored.arguments) == len(cmd.arguments)


# ── engine walker tests with real flowchart files ─────────────────────

import asyncio

from flowcoder_engine.session import QueryResult, TokenUsage
from flowcoder_engine.walker import GraphWalker


class _MockSession:
    """Inline mock session for e2e tests."""

    def __init__(self, responses=None):
        self.name = "mock"
        self.session_id = "mock"
        self.total_cost = 0.0
        self.token_usage = TokenUsage()
        self._responses = list(responses or ["OK"])
        self._i = 0
        self._clear_count = 0

    async def query(self, prompt, block_id="", block_name=""):
        idx = min(self._i, len(self._responses) - 1)
        self._i += 1
        return QueryResult(response_text=self._responses[idx])

    async def clear(self):
        self._clear_count += 1

    async def stop(self):
        pass

    @property
    def is_running(self):
        return True


class _MockProtocol:
    def __init__(self):
        self.messages = []
        self.logs = []

    def emit(self, msg): self.messages.append(msg)
    def emit_block_start(self, *a): pass
    def emit_block_complete(self, *a, **kw): pass
    def emit_result(self, *a, **kw): pass
    def emit_forwarded(self, *a, **kw): pass
    def emit_system(self, *a, **kw): pass
    def emit_stderr(self, *a, **kw): pass
    def log(self, msg): self.logs.append(msg)
    async def start(self): pass
    async def stop(self): pass


def test_execute_simple_example():
    cmd = load_command(EXAMPLES / "simple.json")
    session = _MockSession(["Hello from Claude!"])
    proto = _MockProtocol()
    walker = GraphWalker(cmd.flowchart, session, {"$1": "Hello"}, proto)
    result = asyncio.run(walker.run())
    assert result.status == "completed"
    assert len(result.log) == 3  # start, prompt, end


def test_execute_multi_session_approved():
    """Multi-session uses single session now; all prompts go through it."""
    cmd = load_command(EXAMPLES / "multi_session.json")
    # All prompts use the same session, responses are consumed in order
    session = _MockSession([
        "def hello(): print('hi')",
        '{"approved": true, "feedback": "Looks great!"}',
    ])
    proto = _MockProtocol()
    walker = GraphWalker(
        cmd.flowchart, session, {"$1": "hello function"}, proto
    )
    result = asyncio.run(walker.run())
    assert result.status == "completed"
    assert result.variables.get("approved") is True


def test_execute_multi_session_rejected_then_approved():
    """Multi-session loop: single session handles all prompt blocks in order."""
    cmd = load_command(EXAMPLES / "multi_session.json")
    session = _MockSession([
        "def hello(): print('hi')",  # coder first attempt
        '{"approved": false, "feedback": "Needs docstring"}',  # reviewer rejects
        "def hello():\n    print('hello')",  # coder revised
        '{"approved": true, "feedback": "Good now"}',  # reviewer approves
    ])
    proto = _MockProtocol()
    walker = GraphWalker(
        cmd.flowchart, session, {"$1": "hello function"}, proto
    )
    result = asyncio.run(walker.run())
    assert result.status == "completed"
    assert result.variables.get("approved") is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
