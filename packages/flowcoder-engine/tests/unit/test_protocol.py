"""Tests for ProtocolHandler output methods."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch

import pytest

from flowcoder_engine.protocol import ProtocolHandler


class TestEmit:
    def test_emit_writes_json_line(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit({"type": "test", "data": 42})
        line = captured.getvalue().strip()
        assert json.loads(line) == {"type": "test", "data": 42}

    def test_emit_system(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_system("block_start", {"block_id": "b1"})
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "system"
        assert msg["subtype"] == "block_start"
        assert msg["data"]["block_id"] == "b1"

    def test_emit_flowchart_start(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_flowchart_start("story", "dragons", 5)
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "system"
        assert msg["subtype"] == "flowchart_start"
        assert msg["data"]["command"] == "story"
        assert msg["data"]["block_count"] == 5

    def test_emit_flowchart_complete(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_flowchart_complete("completed", duration_ms=1000, cost_usd=0.05, blocks_executed=3)
        msg = json.loads(captured.getvalue().strip())
        assert msg["subtype"] == "flowchart_complete"
        assert msg["data"]["status"] == "completed"

    def test_emit_forwarded_with_provenance(self):
        """emit_forwarded wraps the inner message with session/block context."""
        p = ProtocolHandler()
        inner = {"type": "assistant", "message": {"content": "hello"}}
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_forwarded(inner, "main", "b1", "Block1")
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "system"
        assert msg["subtype"] == "session_message"
        assert msg["data"]["session"] == "main"
        assert msg["data"]["block_id"] == "b1"
        assert msg["data"]["block_name"] == "Block1"
        assert msg["data"]["message"] == inner

    def test_emit_stderr(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_stderr("error line", "worker-1")
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "system"
        assert msg["subtype"] == "stderr"
        assert msg["data"]["session"] == "worker-1"
        assert msg["data"]["line"] == "error line"

    def test_emit_result(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_result("done", is_error=False, duration_ms=500)
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "result"
        assert msg["subtype"] == "success"
        assert msg["result"] == "done"
        assert msg["is_error"] is False
        assert "uuid" in msg and len(msg["uuid"]) == 32

    def test_emit_block_timeout(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_block_timeout(
                block_id="b1",
                block_name="Slow",
                block_type="prompt",
                elapsed_ms=1234,
                timeout_seconds=1,
            )
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "system"
        assert msg["subtype"] == "block_timeout"
        assert msg["data"] == {
            "block_id": "b1",
            "block_name": "Slow",
            "block_type": "prompt",
            "elapsed_ms": 1234,
            "timeout_seconds": 1,
            "session": "",
        }

    def test_emit_block_complete_with_session_id(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_block_complete("b1", "Slow", True, session_id="sess-abc")
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "system"
        assert msg["subtype"] == "block_complete"
        assert msg["data"]["block_id"] == "b1"
        assert msg["data"]["block_name"] == "Slow"
        assert msg["data"]["success"] is True
        assert msg["data"]["session_id"] == "sess-abc"

    def test_emit_block_complete_omits_session_id_when_none(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_block_complete("b1", "Slow", True)
        msg = json.loads(captured.getvalue().strip())
        assert msg["data"] == {
            "block_id": "b1",
            "block_name": "Slow",
            "success": True,
            "session": "",
        }
        assert "session_id" not in msg["data"]


class TestSessionTagging:
    """Lifecycle events carry the emitting walker's session name."""

    def test_block_start_payload_includes_session(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_block_start("b1", "Prompt", "prompt", session="lint")
        msg = json.loads(captured.getvalue().strip())
        assert msg["subtype"] == "block_start"
        assert msg["data"]["session"] == "lint"

    def test_block_complete_payload_includes_session(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_block_complete("b1", "Prompt", True, session="lint")
        msg = json.loads(captured.getvalue().strip())
        assert msg["subtype"] == "block_complete"
        assert msg["data"]["session"] == "lint"

    def test_block_timeout_payload_includes_session(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_block_timeout("b1", "Slow", "prompt", 1234, 1, session="lint")
        msg = json.loads(captured.getvalue().strip())
        assert msg["subtype"] == "block_timeout"
        assert msg["data"]["session"] == "lint"

    def test_flowchart_start_payload_includes_session(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_flowchart_start("story", "dragons", 5, session="main")
        msg = json.loads(captured.getvalue().strip())
        assert msg["subtype"] == "flowchart_start"
        assert msg["data"]["session"] == "main"

    def test_flowchart_complete_payload_includes_session(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_flowchart_complete("completed", session="main")
        msg = json.loads(captured.getvalue().strip())
        assert msg["subtype"] == "flowchart_complete"
        assert msg["data"]["session"] == "main"

    def test_block_start_session_defaults_to_empty(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_block_start("b1", "Prompt", "prompt")
        msg = json.loads(captured.getvalue().strip())
        assert msg["data"]["session"] == ""


class TestSpawnEvents:
    def test_spawn_start_payload(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_spawn_start(
                agent_name="lint",
                command_name="lint-fix",
                model="opus",
                backend="claude",
                cwd="/tmp",
                parent_session="main",
                session="main",
            )
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "system"
        assert msg["subtype"] == "spawn_start"
        assert msg["data"]["agent_name"] == "lint"
        assert msg["data"]["command_name"] == "lint-fix"
        assert msg["data"]["model"] == "opus"
        assert msg["data"]["backend"] == "claude"
        assert msg["data"]["cwd"] == "/tmp"
        assert msg["data"]["parent_session"] == "main"
        assert msg["data"]["session"] == "main", "emitting (parent) session"

    def test_spawn_start_defaults(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_spawn_start(agent_name="lint")
        msg = json.loads(captured.getvalue().strip())
        assert msg["data"]["command_name"] == ""
        assert msg["data"]["parent_session"] == "main"
        assert msg["data"]["session"] == ""

    def test_spawn_complete_payload(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_spawn_complete(
                agent_name="lint",
                status="completed",
                duration_ms=1234,
                cost_usd=0.042,
                result="{}",
                session="main",
            )
        msg = json.loads(captured.getvalue().strip())
        assert msg["type"] == "system"
        assert msg["subtype"] == "spawn_complete"
        assert msg["data"]["agent_name"] == "lint"
        assert msg["data"]["status"] == "completed"
        assert msg["data"]["duration_ms"] == 1234
        assert msg["data"]["cost_usd"] == 0.042
        assert msg["data"]["result"] == "{}"
        assert msg["data"]["session"] == "main"

    def test_spawn_complete_defaults(self):
        p = ProtocolHandler()
        captured = io.StringIO()
        with patch.object(sys, "stdout", captured):
            p.emit_spawn_complete(agent_name="lint", status="cancelled")
        msg = json.loads(captured.getvalue().strip())
        assert msg["data"]["duration_ms"] == 0
        assert msg["data"]["cost_usd"] == 0.0
        assert msg["data"]["result"] == ""
        assert msg["data"]["session"] == ""


class TestInboxQueue:
    @pytest.mark.asyncio
    async def test_push_and_read(self):
        p = ProtocolHandler()
        p.push_message({"type": "user", "message": "hello"})
        msg = await p.read_message()
        assert msg == {"type": "user", "message": "hello"}

    @pytest.mark.asyncio
    async def test_push_multiple(self):
        p = ProtocolHandler()
        p.push_message({"type": "a"})
        p.push_message({"type": "b"})
        m1 = await p.read_message()
        m2 = await p.read_message()
        assert m1["type"] == "a"
        assert m2["type"] == "b"

    def test_busy_flag_default(self):
        p = ProtocolHandler()
        assert p.busy is False
