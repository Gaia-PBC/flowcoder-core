"""Tests for Session extensions: clone() and stderr forwarding."""

from __future__ import annotations

import asyncio

import pytest
from flowcoder_engine.session import ClaudeSession as Session, _clean_env, _strip_continuity_flags
from tests.conftest import MockProtocol


class TestWithEnv:
    def test_with_env_merges_overrides(self):
        proto = MockProtocol()
        original = Session(
            name="main",
            claude_cmd=["claude"],
            protocol=proto,
            env_overrides={"ANTHROPIC_BASE_URL": "http://parent"},
        )
        child = original.with_env({"ANTHROPIC_BASE_URL": "http://child", "ANTHROPIC_MODEL": "m1"})
        assert child._env_overrides == {
            "ANTHROPIC_BASE_URL": "http://child",
            "ANTHROPIC_MODEL": "m1",
        }
        # Original untouched
        assert original._env_overrides == {"ANTHROPIC_BASE_URL": "http://parent"}

    def test_with_env_on_no_overrides(self):
        original = Session(name="main", claude_cmd=["claude"])
        child = original.with_env({"ANTHROPIC_MODEL": "m1"})
        assert child._env_overrides == {"ANTHROPIC_MODEL": "m1"}

    def test_clean_env_empty_override_unsets_the_variable(self):
        """A child must be able to route BACK to native, not only to a gateway.

        with_env can only ever set, so a spawn pinned to an Anthropic model
        under a parent routed at a local endpoint would inherit the parent's
        ANTHROPIC_BASE_URL and be sent somewhere that does not serve it. An
        empty override value means unset."""
        import os

        os.environ["ANTHROPIC_BASE_URL"] = "http://parent-gateway"
        try:
            env = _clean_env({"ANTHROPIC_BASE_URL": ""})
            assert "ANTHROPIC_BASE_URL" not in env
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)

    def test_clean_env_empty_override_is_a_noop_when_var_absent(self):
        import os

        os.environ.pop("ANTHROPIC_MODEL", None)
        env = _clean_env({"ANTHROPIC_MODEL": ""})
        assert "ANTHROPIC_MODEL" not in env

    def test_clean_env_applies_overrides(self):
        env = _clean_env({"ANTHROPIC_BASE_URL": "http://x"})
        assert env["ANTHROPIC_BASE_URL"] == "http://x"
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"


class TestClone:
    def test_clone_preserves_config(self):
        proto = MockProtocol()
        original = Session(
            name="original",
            claude_cmd=["claude", "--model", "opus"],
            protocol=proto,
        )
        cloned = original.clone("worker-1")

        assert cloned.name == "worker-1"
        assert cloned._claude_cmd == ["claude", "--model", "opus"]
        assert cloned._protocol is proto

    def test_clone_independent_state(self):
        original = Session(name="original", claude_cmd=["claude"])
        original._total_cost = 5.0
        original._session_id = "abc"

        cloned = original.clone("worker")
        assert cloned.total_cost == 0.0
        assert cloned.session_id is None

    def test_clone_does_not_share_cmd_list(self):
        original = Session(name="original", claude_cmd=["claude", "--verbose"])
        cloned = original.clone("worker")

        cloned._claude_cmd.append("--extra")
        assert "--extra" not in original._claude_cmd

    def test_clone_with_control_callback(self):
        async def cb(req):
            return {"type": "control_response"}

        original = Session(
            name="original",
            claude_cmd=["claude"],
            control_callback=cb,
        )
        cloned = original.clone("worker")
        assert cloned._control_callback is cb


class TestWithModel:
    def test_with_model_appends_flag(self):
        """with_model() adds --model to a session that has none."""
        session = Session(name="test", claude_cmd=["claude", "-p"])
        new = session.with_model("haiku")
        assert new._claude_cmd == ["claude", "-p", "--model", "haiku"]
        assert new.name == "test"

    def test_with_model_replaces_existing(self):
        """with_model() replaces an existing --model flag."""
        session = Session(name="test", claude_cmd=["claude", "--model", "opus", "-p"])
        new = session.with_model("sonnet")
        assert new._claude_cmd == ["claude", "--model", "sonnet", "-p"]

    def test_with_model_does_not_mutate_original(self):
        """with_model() returns a new session, original unchanged."""
        session = Session(name="test", claude_cmd=["claude", "-p"])
        new = session.with_model("haiku")
        assert "--model" not in session._claude_cmd
        assert new is not session

    def test_with_model_preserves_protocol(self):
        proto = MockProtocol()
        session = Session(name="test", claude_cmd=["claude"], protocol=proto)
        new = session.with_model("haiku")
        assert new._protocol is proto


class TestEnvOverrides:
    def test_clean_env_no_overrides_unchanged(self):
        env = _clean_env()
        assert "ANTHROPIC_BASE_URL" not in env
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"

    def test_clean_env_applies_overrides(self):
        env = _clean_env({"ANTHROPIC_BASE_URL": "http://127.0.0.1:3000"})
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:3000"
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"

    def test_clean_env_overrides_can_replace_sdk_keys(self):
        env = _clean_env({"CLAUDE_CODE_ENTRYPOINT": "custom"})
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "custom"

    def test_session_stores_env_overrides(self):
        s = Session(name="test", claude_cmd=["claude"], env_overrides={"X": "1"})
        assert s._env_overrides == {"X": "1"}

    def test_session_env_overrides_default_none(self):
        s = Session(name="test", claude_cmd=["claude"])
        assert s._env_overrides is None

    def test_session_env_overrides_copied_not_shared(self):
        overrides = {"X": "1"}
        s = Session(name="test", claude_cmd=["claude"], env_overrides=overrides)
        overrides["X"] = "2"
        assert s._env_overrides == {"X": "1"}

    def test_clone_preserves_env_overrides(self):
        s = Session(name="orig", claude_cmd=["claude"], env_overrides={"X": "1"})
        c = s.clone("worker")
        assert c._env_overrides == {"X": "1"}

    def test_with_model_preserves_env_overrides(self):
        s = Session(name="t", claude_cmd=["claude"], env_overrides={"X": "1"})
        n = s.with_model("haiku")
        assert n._env_overrides == {"X": "1"}


class TestCwd:
    def test_cwd_default_none(self):
        s = Session(name="test", claude_cmd=["claude"])
        assert s._cwd is None

    def test_cwd_stored(self):
        s = Session(name="test", claude_cmd=["claude"], cwd="/tmp/work")
        assert s._cwd == "/tmp/work"

    def test_clone_preserves_cwd(self):
        s = Session(name="orig", claude_cmd=["claude"], cwd="/tmp/work")
        c = s.clone("worker")
        assert c._cwd == "/tmp/work"

    def test_with_model_preserves_cwd(self):
        s = Session(name="t", claude_cmd=["claude"], cwd="/tmp/work")
        n = s.with_model("haiku")
        assert n._cwd == "/tmp/work"


class TestStreamQuery:
    @pytest.mark.asyncio
    async def test_stream_query_yields_messages_then_result(self):
        """stream_query yields each subprocess message; result updates state."""
        from unittest.mock import AsyncMock

        s = Session(name="test", claude_cmd=["claude"])

        # Fake process: yields system, assistant, result, then EOF
        messages = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {
                "type": "result",
                "total_cost_usd": 0.05,
                "session_id": "sess-123",
                "duration_ms": 500,
            },
        ]

        fake_process = AsyncMock()
        fake_process.write = AsyncMock()
        read_iter = iter(messages)
        async def _read(timeout=None):
            try:
                return next(read_iter)
            except StopIteration:
                return None
        fake_process.read = _read
        s._process = fake_process

        collected = []
        async for chunk in s.stream_query("test prompt"):
            collected.append(chunk)

        assert len(collected) == 3
        assert collected[0]["type"] == "system"
        assert collected[1]["type"] == "assistant"
        assert collected[2]["type"] == "result"
        assert s.session_id == "sess-123"
        assert s.total_cost == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_stream_query_filters_rate_limit(self):
        """rate_limit_event is filtered out before yielding."""
        from unittest.mock import AsyncMock

        s = Session(name="test", claude_cmd=["claude"])

        messages = [
            {"type": "rate_limit_event"},
            {"type": "assistant", "message": {"content": []}},
            {"type": "result", "total_cost_usd": 0.0},
        ]

        fake_process = AsyncMock()
        fake_process.write = AsyncMock()
        read_iter = iter(messages)
        async def _read(timeout=None):
            try:
                return next(read_iter)
            except StopIteration:
                return None
        fake_process.read = _read
        s._process = fake_process

        collected = [c async for c in s.stream_query("test")]
        types = [c["type"] for c in collected]
        assert "rate_limit_event" not in types
        assert types == ["assistant", "result"]


class TestStderrForwarding:
    @pytest.mark.asyncio
    async def test_stderr_forwarded_to_protocol(self):
        proto = MockProtocol()
        session = Session(name="test", claude_cmd=["echo"], protocol=proto)

        # Simulate: create a process that writes to stderr
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", "echo 'error line 1' >&2; echo 'error line 2' >&2",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Manually wire up the session's stderr forwarding
        from flowcoder_engine.subprocess import ClaudeProcess

        cp = ClaudeProcess()
        cp._proc = proc
        session._process = cp
        session._stderr_task = asyncio.create_task(session._forward_stderr())

        await asyncio.wait_for(session._stderr_task, timeout=5.0)

        assert len(proto.stderr_lines) == 2
        assert proto.stderr_lines[0] == {"session": "test", "line": "error line 1"}
        assert proto.stderr_lines[1] == {"session": "test", "line": "error line 2"}

        await cp.stop()

    @pytest.mark.asyncio
    async def test_stderr_without_protocol_logs(self):
        session = Session(name="test", claude_cmd=["echo"])

        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", "echo 'debug msg' >&2",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        from flowcoder_engine.subprocess import ClaudeProcess

        cp = ClaudeProcess()
        cp._proc = proc
        session._process = cp
        session._stderr_task = asyncio.create_task(session._forward_stderr())

        # Should complete without error (logs to debug instead of protocol)
        await asyncio.wait_for(session._stderr_task, timeout=5.0)
        await cp.stop()


class TestRefreshStripsContinuityFlags:
    """A refresh (clear()) must not respawn with --resume/--continue, or the
    'cleared' session reloads the prior conversation (the flowcoder refresh bug).
    """

    def test_strip_removes_resume_and_value(self):
        assert _strip_continuity_flags(
            ["claude", "-p", "--model", "haiku", "--resume", "abc123"]
        ) == ["claude", "-p", "--model", "haiku"]

    def test_strip_removes_continue_and_session_id(self):
        assert _strip_continuity_flags(
            ["claude", "--continue", "--session-id", "sid-1", "-p"]
        ) == ["claude", "-p"]

    def test_strip_bare_resume_without_value(self):
        # --resume immediately followed by another flag: drop only the flag.
        assert _strip_continuity_flags(
            ["claude", "--resume", "--model", "haiku"]
        ) == ["claude", "--model", "haiku"]

    def test_strip_is_noop_without_continuity_flags(self):
        cmd = ["claude", "-p", "--model", "haiku"]
        assert _strip_continuity_flags(cmd) == cmd

    async def test_clear_drops_resume_from_respawn_cmd(self):
        # Regression guard for the refresh bug: after clear(), the session must
        # NOT respawn with --resume (which would reload the prior conversation).
        session = Session("t", ["claude", "-p", "--model", "haiku", "--resume", "abc123"])
        session._session_id = "abc123"
        restarts: list[bool] = []

        async def fake_stop() -> None:
            pass

        async def fake_start() -> None:
            restarts.append(True)

        session.stop = fake_stop  # type: ignore[method-assign]
        session.start = fake_start  # type: ignore[method-assign]

        await session.clear()

        assert "--resume" not in session._claude_cmd
        assert "abc123" not in session._claude_cmd
        assert session._session_id is None
        assert restarts == [True]  # it did restart, just without --resume
