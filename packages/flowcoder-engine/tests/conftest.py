"""Shared fixtures for engine tests."""

from __future__ import annotations

import asyncio

import pytest
from flowcoder_engine.session import BaseSession, QueryResult, TokenUsage
from flowcoder_flowchart import (
    BashBlock,
    BranchBlock,
    CommandBlock,
    Connection,
    EndBlock,
    Flowchart,
    PromptBlock,
    RefreshBlock,
    SessionConfig,
    StartBlock,
    VariableBlock,
    VariableType,
)


class MockSession(BaseSession):
    """Mock session for unit testing the walker."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        delay_seconds: float = 0.0,
        session_id: str | None = "mock-session",
    ):
        self._name = "mock"
        self._session_id: str | None = session_id
        self._total_cost = 0.0
        self._token_usage = TokenUsage()
        self._clone_cwd: str | None = None
        self._responses = list(responses or ["Mock response"])
        self._call_count = 0
        self._clear_count = 0
        self._delay_seconds = delay_seconds
        self._stop_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def token_usage(self) -> TokenUsage:
        return self._token_usage

    @property
    def is_running(self) -> bool:
        return True

    def clone(self, name: str, cwd: str | None = None) -> MockSession:
        ms = MockSession(
            responses=list(self._responses),
            delay_seconds=self._delay_seconds,
            session_id=self._session_id,
        )
        ms._name = name
        # Record the cwd the walker cloned with, so spawn-cwd wiring is testable.
        ms._clone_cwd = cwd
        return ms

    def with_model(self, model: str) -> MockSession:
        return self.clone(self._name)

    async def start(self) -> None:
        pass

    async def query(
        self, prompt: str, block_id: str = "", block_name: str = ""
    ) -> QueryResult:
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        idx = min(self._call_count, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_count += 1
        return QueryResult(response_text=text, cost_usd=0.01)

    async def clear(self) -> None:
        """Mock clear — resets call count to simulate fresh history."""
        self._clear_count += 1

    async def stop(self) -> None:
        self._stop_count += 1


class MockProtocol:
    """Mock protocol handler that records emitted messages."""

    def __init__(self):
        self.messages: list[dict] = []
        self.logs: list[str] = []
        self.forwarded: list[dict] = []
        self.stderr_lines: list[dict] = []

    def emit(self, msg: dict) -> None:
        self.messages.append(msg)

    def emit_system(self, subtype: str, data: dict | None = None) -> None:
        msg: dict = {"type": "system", "subtype": subtype}
        if data:
            msg["data"] = data
        self.messages.append(msg)

    def emit_block_start(
        self, block_id: str, block_name: str, block_type: str, session: str = ""
    ) -> None:
        self.messages.append({
            "type": "system",
            "subtype": "block_start",
            "data": {
                "block_id": block_id,
                "block_name": block_name,
                "block_type": block_type,
                "session": session,
            },
        })

    def emit_block_complete(
        self,
        block_id: str,
        block_name: str,
        success: bool,
        session_id: str | None = None,
        session: str = "",
    ) -> None:
        data: dict = {
            "block_id": block_id,
            "block_name": block_name,
            "success": success,
            "session": session,
        }
        if session_id is not None:
            data["session_id"] = session_id
        self.messages.append({
            "type": "system",
            "subtype": "block_complete",
            "data": data,
        })

    def emit_block_timeout(
        self,
        block_id: str,
        block_name: str,
        block_type: str,
        elapsed_ms: int,
        timeout_seconds: int,
        session: str = "",
    ) -> None:
        self.messages.append({
            "type": "system",
            "subtype": "block_timeout",
            "data": {
                "block_id": block_id,
                "block_name": block_name,
                "block_type": block_type,
                "elapsed_ms": elapsed_ms,
                "timeout_seconds": timeout_seconds,
                "session": session,
            },
        })

    def emit_spawn_start(
        self,
        *,
        agent_name: str,
        command_name: str = "",
        model: str = "",
        backend: str = "",
        cwd: str = "",
        parent_session: str = "",
        session: str = "",
    ) -> None:
        self.messages.append({
            "type": "system",
            "subtype": "spawn_start",
            "data": {
                "agent_name": agent_name,
                "command_name": command_name,
                "model": model,
                "backend": backend,
                "cwd": cwd,
                "parent_session": parent_session,
                "session": session,
            },
        })

    def emit_spawn_complete(
        self,
        *,
        agent_name: str,
        status: str,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
        result: str = "",
        session: str = "",
    ) -> None:
        self.messages.append({
            "type": "system",
            "subtype": "spawn_complete",
            "data": {
                "agent_name": agent_name,
                "status": status,
                "duration_ms": duration_ms,
                "cost_usd": cost_usd,
                "result": result,
                "session": session,
            },
        })

    def emit_result(self, *args, **kwargs) -> None:
        self.messages.append({"type": "result", "args": args, "kwargs": kwargs})

    def emit_forwarded(
        self,
        inner_msg: dict,
        session_name: str,
        block_id: str,
        block_name: str,
    ) -> None:
        self.forwarded.append({
            "session": session_name,
            "block_id": block_id,
            "block_name": block_name,
            "message": inner_msg,
        })

    def emit_stderr(self, line: str, session_name: str) -> None:
        self.stderr_lines.append({"session": session_name, "line": line})

    def log(self, message: str) -> None:
        self.logs.append(message)


@pytest.fixture
def simple_flowchart() -> Flowchart:
    """start -> prompt -> end"""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "p": PromptBlock(id="p", name="Ask", prompt="Hello $1"),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="p"),
            Connection(source_id="p", target_id="e"),
        ],
    )


@pytest.fixture
def branch_flowchart() -> Flowchart:
    """start -> branch -> (true: end_ok, false: end_fail)"""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "b": BranchBlock(id="b", name="Check", condition="flag"),
            "ok": EndBlock(id="ok", name="OK"),
            "fail": EndBlock(id="fail", name="Fail"),
        },
        connections=[
            Connection(source_id="s", target_id="b"),
            Connection(source_id="b", target_id="ok", is_true_path=True),
            Connection(source_id="b", target_id="fail", is_true_path=False),
        ],
    )


@pytest.fixture
def variable_flowchart() -> Flowchart:
    """start -> set_var -> prompt (using var) -> end"""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "v": VariableBlock(
                id="v", name="Set Name",
                variable_name="name", variable_value="World",
            ),
            "p": PromptBlock(id="p", name="Greet", prompt="Hello {{name}}"),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="v"),
            Connection(source_id="v", target_id="p"),
            Connection(source_id="p", target_id="e"),
        ],
    )


@pytest.fixture
def bash_flowchart() -> Flowchart:
    """start -> bash -> end"""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "b": BashBlock(
                id="b", name="Run",
                command="echo hello",
                capture_output=True,
                output_variable="result",
            ),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="b"),
            Connection(source_id="b", target_id="e"),
        ],
    )


@pytest.fixture
def multi_session_flowchart() -> Flowchart:
    """Two sessions: deployer and reviewer (both use main session now)."""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "d": PromptBlock(id="d", name="Deploy", prompt="Deploy $1", session="deployer"),
            "r": PromptBlock(id="r", name="Review", prompt="Review deployment", session="reviewer"),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="d"),
            Connection(source_id="d", target_id="r"),
            Connection(source_id="r", target_id="e"),
        ],
        sessions={
            "deployer": SessionConfig(model="opus"),
            "reviewer": SessionConfig(model="sonnet"),
        },
    )


@pytest.fixture
def refresh_flowchart() -> Flowchart:
    """start -> prompt -> refresh -> prompt -> end"""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "p1": PromptBlock(id="p1", name="Ask1", prompt="First"),
            "rf": RefreshBlock(id="rf", name="Reset", target_session="default"),
            "p2": PromptBlock(id="p2", name="Ask2", prompt="Second"),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="p1"),
            Connection(source_id="p1", target_id="rf"),
            Connection(source_id="rf", target_id="p2"),
            Connection(source_id="p2", target_id="e"),
        ],
    )


@pytest.fixture
def loop_flowchart() -> Flowchart:
    """start -> set_counter -> prompt -> inc_counter -> branch -> (true: end, false: prompt loop)"""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "init": VariableBlock(
                id="init", name="Init",
                variable_name="counter", variable_value="0",
                variable_type=VariableType.NUMBER,
            ),
            "p": PromptBlock(id="p", name="Work", prompt="Iteration {{counter}}"),
            "inc": VariableBlock(
                id="inc", name="Increment",
                variable_name="done", variable_value="true",
                variable_type=VariableType.BOOLEAN,
            ),
            "check": BranchBlock(id="check", name="Done?", condition="done"),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="init"),
            Connection(source_id="init", target_id="p"),
            Connection(source_id="p", target_id="inc"),
            Connection(source_id="inc", target_id="check"),
            Connection(source_id="check", target_id="e", is_true_path=True),
            Connection(source_id="check", target_id="p", is_true_path=False),
        ],
    )


@pytest.fixture
def command_flowchart() -> Flowchart:
    """start -> command block -> end"""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "c": CommandBlock(
                id="c", name="SubCmd",
                command_name="test-sub",
                arguments="{{task}}",
                inherit_variables=False,
                merge_output=True,
            ),
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="c"),
            Connection(source_id="c", target_id="e"),
        ],
    )


# ── Policy: skips are failures (SOUL prompting fix #5) ────────────────
# A skipped/xfailed test produces zero evidence, so a suite may not
# contain one and still be reported as passing. Convert any skip/xfail
# outcome into a failure. Deselection (e.g. -m "not slow") is untouched:
# deselected tests are never collected and produce no report.
@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.skipped:
        report.outcome = "failed"
        report.longrepr = f"SKIP DISALLOWED by project policy (skips are failures): {report.longrepr}"
    return report
