"""GUIProtocolBridge — routes engine protocol events to GUI callbacks.

Subclasses the engine's ProtocolHandler so GraphWalker can emit events
without knowing about the GUI. Instead of writing JSON to stdout, events
are dispatched to registered callbacks that update the GUI.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

from flowcoder_engine.protocol import ProtocolHandler

from src.models.execution import BlockResult, ExecutionContext, ExecutionStatus

if TYPE_CHECKING:
    from src.models.blocks import Block
    from src.models.flowchart import Flowchart

log = logging.getLogger(__name__)


class GUIProtocolBridge(ProtocolHandler):
    """Routes engine protocol events to GUI callbacks.

    The engine's GraphWalker calls emit methods during execution.
    This bridge intercepts those calls and dispatches them to the
    GUI's callback system instead of writing JSON to stdout.
    """

    def __init__(
        self,
        flowchart: Flowchart,
        context: ExecutionContext,
        *,
        on_block_start: Optional[Callable[[Block, ExecutionContext], None]] = None,
        on_block_complete: Optional[
            Callable[[Block, BlockResult, ExecutionContext], None]
        ] = None,
        on_execution_start: Optional[Callable[[str, ExecutionContext], None]] = None,
        on_execution_complete: Optional[Callable[[ExecutionContext], None]] = None,
        on_prompt_stream: Optional[Callable[[str, str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__()
        self._flowchart = flowchart
        self._context = context
        self._on_block_start = on_block_start
        self._on_block_complete = on_block_complete
        self._on_execution_start = on_execution_start
        self._on_execution_complete = on_execution_complete
        self._on_prompt_stream = on_prompt_stream
        self._on_stderr = on_stderr

    # -- Lifecycle: no stdin/stdout I/O needed --

    async def start(self) -> None:
        """No-op — GUI doesn't use stdin pipes."""

    async def stop(self) -> None:
        """No-op — no stdin task to cancel."""

    # -- Override emit to prevent stdout writes --

    def emit(self, msg: dict[str, Any]) -> None:
        """Route messages to callbacks instead of stdout."""
        msg_type = msg.get("type")
        if msg_type == "result":
            self._handle_result_message(msg)

    # -- Block lifecycle events --

    def emit_block_start(
        self, block_id: str, block_name: str, block_type: str
    ) -> None:
        self._context.current_block_id = block_id
        if self._on_block_start:
            block = self._flowchart.blocks.get(block_id)
            if block:
                self._on_block_start(block, self._context)

    def emit_block_complete(
        self,
        block_id: str,
        block_name: str,
        success: bool,
        session_id: str | None = None,
    ) -> None:
        # session_id is ignored: it exists so out-of-process drivers can keep
        # their stored id in sync, but the GUI already owns the session object
        # and reads the id from GUISessionAdapter.  Same as
        # emit_flowchart_complete below.
        if self._on_block_complete:
            block = self._flowchart.blocks.get(block_id)
            if block:
                result = BlockResult(success=success)
                self._on_block_complete(block, result, self._context)

    # -- Flowchart lifecycle events --

    def emit_flowchart_start(
        self, command: str, args: str, block_count: int
    ) -> None:
        self._context.status = ExecutionStatus.RUNNING
        self._context.start_time = datetime.now()
        if self._on_execution_start:
            self._on_execution_start(command, self._context)

    def emit_flowchart_complete(
        self,
        status: str,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
        blocks_executed: int = 0,
        session_id: str = "",
    ) -> None:
        exec_status = (
            ExecutionStatus.COMPLETED if status == "success" else ExecutionStatus.ERROR
        )
        self._context.complete(exec_status)
        if self._on_execution_complete:
            self._on_execution_complete(self._context)

    # -- Streaming / forwarded messages --

    def emit_forwarded(
        self,
        inner_msg: dict[str, Any],
        session_name: str,
        block_id: str,
        block_name: str,
    ) -> None:
        """Route inner session messages to the prompt stream callback."""
        if not self._on_prompt_stream:
            return

        # Extract text content from the inner claude message
        content = _extract_text(inner_msg)
        if content:
            self._on_prompt_stream(block_name, content)

    def emit_stderr(self, line: str, session_name: str) -> None:
        if self._on_stderr:
            self._on_stderr(line)

    # -- Logging: use Python logging instead of stderr --

    def log(self, message: str) -> None:
        log.debug("[engine] %s", message)

    # -- Result messages --

    def _handle_result_message(self, msg: dict[str, Any]) -> None:
        """Handle result messages (turn completions from the engine)."""
        result_text = msg.get("result", "")
        if self._on_prompt_stream and result_text:
            block_name = self._context.current_block_id or ""
            self._on_prompt_stream(block_name, result_text)

    # -- Control forwarding (not needed in GUI) --

    async def forward_control_request(
        self, inner_request: dict[str, Any]
    ) -> dict[str, Any]:
        """Not applicable in GUI mode — permissions are handled by the service."""
        raise NotImplementedError(
            "GUI mode handles permissions via ClaudeEngineService, "
            "not protocol-level control forwarding."
        )


def _extract_text(inner_msg: dict[str, Any]) -> str:
    """Extract displayable text from an inner claude message.

    The inner message may be a Claude SDK JSON message with various
    content types. We extract text blocks for display.
    """
    # Direct text content
    if isinstance(inner_msg.get("content"), str):
        return inner_msg["content"]

    # Content blocks array (Claude SDK format)
    content = inner_msg.get("content")
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "".join(texts)

    # Result field (used in emit_result messages)
    if "result" in inner_msg:
        return str(inner_msg["result"])

    return ""
