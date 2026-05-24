"""Protocol handler for stdout JSON-lines output.

Writes stream-json messages to stdout.  Used by the proxy core
and by Session/GraphWalker during flowchart execution.  All output
goes to stdout; log messages go to stderr.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Any


class ProtocolHandler:
    """Reads from stdin, writes to stdout. JSON-lines protocol."""

    def __init__(self) -> None:
        self._stdin_reader: asyncio.StreamReader | None = None
        self._pending_control: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._stdin_task: asyncio.Task[None] | None = None
        self._inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.busy: bool = False

    async def start(self) -> None:
        """Set up async stdin reader and start routing messages."""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
        self._stdin_reader = reader
        self._stdin_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """Read stdin lines, route control responses to futures, rest to inbox."""
        assert self._stdin_reader is not None
        while True:
            line = await self._stdin_reader.readline()
            if not line:
                break

            text = line.decode().strip()
            if not text:
                continue

            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "control_response":
                req_id = msg.get("response", {}).get("request_id", "")
                if req_id in self._pending_control:
                    self._pending_control[req_id].set_result(msg)
            elif msg_type == "status_request":
                self.emit({"type": "status_response", "busy": self.busy})
            else:
                await self._inbox.put(msg)

    async def read_message(self) -> dict[str, Any]:
        """Read next message from inbox (user, command, shutdown, etc.)."""
        return await self._inbox.get()

    def push_message(self, msg: dict[str, Any]) -> None:
        """Push a message directly into the inbox queue."""
        self._inbox.put_nowait(msg)

    async def forward_control_request(
        self, inner_request: dict[str, Any]
    ) -> dict[str, Any]:
        """Forward a control request from inner claude to outer SDK."""
        original_id = inner_request.get("request_id", "")
        outer_id = f"fc_{original_id}"
        inner_request["request_id"] = outer_id

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_control[outer_id] = future

        self.emit(inner_request)
        response = await future
        del self._pending_control[outer_id]

        if "response" in response:
            response["response"]["request_id"] = original_id

        return response

    async def stop(self) -> None:
        """Cancel the stdin read loop."""
        if self._stdin_task:
            self._stdin_task.cancel()
            try:
                await self._stdin_task
            except asyncio.CancelledError:
                pass

    def emit(self, msg: dict[str, Any]) -> None:
        """Write a JSON message to stdout (one line)."""
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def emit_system(self, subtype: str, data: dict[str, Any] | None = None) -> None:
        """Emit a system message."""
        msg: dict[str, Any] = {"type": "system", "subtype": subtype}
        if data:
            msg["data"] = data
        self.emit(msg)

    def emit_block_start(self, block_id: str, block_name: str, block_type: str) -> None:
        """Emit block_start system message."""
        self.emit_system(
            "block_start",
            {"block_id": block_id, "block_name": block_name, "block_type": block_type},
        )

    def emit_block_complete(
        self,
        block_id: str,
        block_name: str,
        success: bool,
        session_id: str | None = None,
    ) -> None:
        """Emit block_complete system message.

        ``session_id`` is included in the data payload only when non-null,
        so consumers can update their stored session_id without overwriting
        with a None before the first prompt block has run.
        """
        data: dict[str, Any] = {
            "block_id": block_id,
            "block_name": block_name,
            "success": success,
        }
        if session_id is not None:
            data["session_id"] = session_id
        self.emit_system("block_complete", data)

    def emit_block_timeout(
        self,
        block_id: str,
        block_name: str,
        block_type: str,
        elapsed_ms: int,
        timeout_seconds: int,
    ) -> None:
        """Emit block_timeout system message when a block exceeds its limit."""
        self.emit_system(
            "block_timeout",
            {
                "block_id": block_id,
                "block_name": block_name,
                "block_type": block_type,
                "elapsed_ms": elapsed_ms,
                "timeout_seconds": timeout_seconds,
            },
        )

    def emit_flowchart_start(
        self, command: str, args: str, block_count: int
    ) -> None:
        """Emit flowchart_start when entering takeover mode."""
        self.emit_system(
            "flowchart_start",
            {"command": command, "args": args, "block_count": block_count},
        )

    def emit_flowchart_complete(
        self,
        status: str,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
        blocks_executed: int = 0,
        session_id: str = "",
    ) -> None:
        """Emit flowchart_complete when leaving takeover mode."""
        self.emit_system(
            "flowchart_complete",
            {
                "status": status,
                "duration_ms": duration_ms,
                "cost_usd": cost_usd,
                "blocks_executed": blocks_executed,
                "session_id": session_id,
            },
        )

    def emit_result(
        self,
        result_text: str,
        is_error: bool = False,
        duration_ms: int = 0,
        num_turns: int = 0,
        total_cost_usd: float = 0.0,
        session_id: str = "flowchart",
    ) -> None:
        """Emit a result message (turn completion)."""
        self.emit(
            {
                "type": "result",
                "subtype": "error" if is_error else "success",
                "uuid": uuid.uuid4().hex,
                "session_id": session_id,
                "duration_ms": duration_ms,
                "duration_api_ms": 0,
                "is_error": is_error,
                "num_turns": num_turns,
                "total_cost_usd": total_cost_usd,
                "result": result_text,
            }
        )

    def emit_forwarded(
        self,
        inner_msg: dict[str, Any],
        session_name: str,
        block_id: str,
        block_name: str,
    ) -> None:
        """Forward an inner claude message to outer SDK, tagged with context."""
        self.emit({
            "type": "system",
            "subtype": "session_message",
            "data": {
                "session": session_name,
                "block_id": block_id,
                "block_name": block_name,
                "message": inner_msg,
            },
        })

    def emit_stderr(self, line: str, session_name: str) -> None:
        """Forward an inner claude stderr line to the client."""
        self.emit_system("stderr", {"session": session_name, "line": line})

    def log(self, message: str) -> None:
        """Write a log line to stderr."""
        sys.stderr.write(f"[flowcoder] {message}\n")
        sys.stderr.flush()
