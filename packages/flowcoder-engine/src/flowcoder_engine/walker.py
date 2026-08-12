"""Graph walker — executes a flowchart by walking blocks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import time
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from flowcoder_flowchart import (
    BashBlock,
    BlockBase,
    BlockType,
    BranchBlock,
    CommandBlock,
    ExitBlock,
    Flowchart,
    InputBlock,
    PromptBlock,
    RefreshBlock,
    SpawnBlock,
    VariableBlock,
    VariableType,
    WaitBlock,
)
from opentelemetry import trace

from .json_parser import parse_json_from_response
from .resolver import CommandNotFoundError, resolve_command
from .subprocess import ReadInterruptedError, ReadTimeoutError
from .templates import evaluate_template

if TYPE_CHECKING:
    from pathlib import Path

    from .protocol import ProtocolHandler
    from .session import BaseSession
    from .session_factory import SessionFactory

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)

SOFT_TIMEOUT_SECONDS = 300  # 5 minutes — log warning, don't kill
MAX_RECURSION_DEPTH = 10


class ExecutionError(Exception):
    pass


@dataclass
class BlockResult:
    success: bool = True
    output: str = ""
    error: str = ""
    branch_taken: bool | None = None
    exit_code: int | None = None

    @classmethod
    def ok(cls, output: str = "", branch_taken: bool | None = None) -> BlockResult:
        return cls(success=True, output=output, branch_taken=branch_taken)

    @classmethod
    def fail(cls, error: str) -> BlockResult:
        return cls(success=False, error=error)

    @classmethod
    def exit(cls, code: int = 0, message: str = "") -> BlockResult:
        return cls(success=True, output=message, exit_code=code)


@dataclass
class LogEntry:
    block_id: str
    block_name: str
    block_type: str
    result: BlockResult
    duration_ms: int = 0


@dataclass
class ExecutionResult:
    variables: dict[str, Any]
    log: list[LogEntry]
    status: str  # "completed" | "halted" | "error" | "exited"
    exit_code: int = 0
    duration_ms: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


_COMPARISON_RE = re.compile(
    r"^\s*(.+?)\s+(==|!=|>=|<=|>|<)\s+(.+?)\s*$"
)


def _is_truthy(value: Any) -> bool:
    """Test truthiness the same way Pride does."""
    if value is None:
        return False
    s = str(value).lower().strip()
    return bool(value) and s not in ("false", "0", "no", "")


def _coerce_numeric(s: str) -> float | str:
    """Try to interpret a string as a number; return as-is on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def _evaluate_condition(condition: str, variables: dict[str, Any]) -> bool:
    """Evaluate a branch condition string.

    Supports:
    - Simple variable truthiness: ``fullyImplemented``
    - Negation: ``!hasErrors``
    - Comparison: ``exitCode == 0``, ``count > 5``, ``status != "done"``

    Template substitution (``{{var}}``, ``$N``) should be done *before*
    calling this function so that comparison values are already resolved.
    """
    cond = condition.strip()

    # Negation: !varname
    if cond.startswith("!"):
        inner = cond[1:].strip()
        return not _is_truthy(variables.get(inner))

    # Comparison operators
    m = _COMPARISON_RE.match(cond)
    if m:
        lhs_raw, op, rhs_raw = m.group(1), m.group(2), m.group(3)

        # Strip surrounding quotes from RHS if present
        rhs_str = rhs_raw.strip("\"'")

        # Resolve LHS from variables (could be a variable name or literal)
        lhs_val = variables.get(lhs_raw, lhs_raw)

        # Coerce both sides for numeric comparison
        lhs_num = _coerce_numeric(str(lhs_val))
        rhs_num = _coerce_numeric(rhs_str)

        # If both are numbers, compare numerically
        if isinstance(lhs_num, float) and isinstance(rhs_num, float):
            match op:
                case "==": return lhs_num == rhs_num
                case "!=": return lhs_num != rhs_num
                case ">":  return lhs_num > rhs_num
                case "<":  return lhs_num < rhs_num
                case ">=": return lhs_num >= rhs_num
                case "<=": return lhs_num <= rhs_num

        # String comparison fallback
        lhs_s = str(lhs_val)
        match op:
            case "==": return lhs_s == rhs_str
            case "!=": return lhs_s != rhs_str
            case ">":  return lhs_s > rhs_str
            case "<":  return lhs_s < rhs_str
            case ">=": return lhs_s >= rhs_str
            case "<=": return lhs_s <= rhs_str

    # Simple variable lookup (original behavior)
    return _is_truthy(variables.get(cond))


class GraphWalker:
    """Walks a flowchart graph, executing blocks."""

    def __init__(
        self,
        flowchart: Flowchart,
        session: BaseSession,
        variables: dict[str, Any],
        protocol: ProtocolHandler,
        max_blocks: int = 1000,
        call_stack: list[str] | None = None,
        max_depth: int = MAX_RECURSION_DEPTH,
        search_paths: list[str | Path] | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._flowchart = flowchart
        self._session = session
        self._variables = variables
        self._protocol = protocol
        self._log: list[LogEntry] = []
        self._halted = False
        self._blocks_executed = 0
        self._max_blocks = max_blocks
        self._call_stack = call_stack or []
        self._max_depth = max_depth
        self._search_paths = search_paths or []
        self._session_factory = session_factory
        self._spawned_sessions: dict[str, BaseSession] = {}
        self._spawned_tasks: dict[str, asyncio.Task[ExecutionResult]] = {}
        self._spawned_results: dict[str, ExecutionResult] = {}
        self._halt_requested = False

    def halt(self) -> None:
        """Request the walker to halt after the current block."""
        self._halt_requested = True

    def resume(self) -> None:
        """Clear the halt flag so execution can continue."""
        self._halt_requested = False
        self._halted = False

    async def run(self) -> ExecutionResult:
        """Execute the flowchart from start to end."""
        with _tracer.start_as_current_span(
            "flowchart.run",
            attributes={
                "flowchart.name": self._flowchart.name,
                "flowchart.block_count": len(self._flowchart.blocks),
            },
        ) as run_span:
            start_time = time.monotonic()
            current = self._find_start_block()
            exit_code = 0

            try:
                while current and not self._halted:
                    if self._halt_requested:
                        self._halted = True
                        break

                    if self._blocks_executed >= self._max_blocks:
                        raise ExecutionError(
                            f"Safety limit: exceeded {self._max_blocks} blocks"
                        )

                    self._blocks_executed += 1
                    self._protocol.emit_block_start(
                        current.id, current.name, current.type
                    )
                    self._protocol.log(
                        f"Executing block \"{current.name}\" "
                        f"({current.type}, session={current.session})"
                    )

                    block_start = time.monotonic()
                    result = await self._execute_block(current)
                    block_ms = int((time.monotonic() - block_start) * 1000)

                    if block_ms > SOFT_TIMEOUT_SECONDS * 1000:
                        self._protocol.log(
                            f"WARNING: Block \"{current.name}\" took {block_ms}ms "
                            f"(>{SOFT_TIMEOUT_SECONDS}s soft timeout)"
                        )

                    entry = LogEntry(
                        block_id=current.id,
                        block_name=current.name,
                        block_type=current.type,
                        result=result,
                        duration_ms=block_ms,
                    )
                    self._log.append(entry)

                    self._protocol.emit_block_complete(
                        current.id,
                        current.name,
                        result.success,
                        session_id=self._session.session_id,
                    )

                    if result.exit_code is not None:
                        exit_code = result.exit_code
                        break

                    if not result.success:
                        self._halted = True
                        break

                    current = self._next_block(current, result)
            finally:
                await self._cleanup_spawned()

            total_ms = int((time.monotonic() - start_time) * 1000)

            if exit_code != 0:
                status = "exited"
            elif self._halted:
                status = "halted"
            else:
                status = "completed"

            run_span.set_attributes({"flowchart.status": status, "flowchart.duration_ms": total_ms})
            usage = self._session.token_usage
            return ExecutionResult(
                variables=self._variables,
                log=self._log,
                status=status,
                exit_code=exit_code,
                duration_ms=total_ms,
                cost_usd=self._session.total_cost,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_tokens=usage.cache_creation_tokens,
                cache_read_tokens=usage.cache_read_tokens,
            )

    async def _execute_block(self, block: BlockBase) -> BlockResult:
        """Dispatch to the appropriate handler based on block type."""
        with _tracer.start_as_current_span(
            "flowchart.block",
            attributes={"block.id": block.id, "block.name": block.name, "block.type": block.type},
        ):
            match block.type:
                case BlockType.START:
                    return BlockResult.ok()
                case BlockType.END:
                    return BlockResult.ok()
                case BlockType.PROMPT:
                    assert isinstance(block, PromptBlock)
                    return await self._exec_prompt(block)
                case BlockType.BRANCH:
                    assert isinstance(block, BranchBlock)
                    return self._exec_branch(block)
                case BlockType.VARIABLE:
                    assert isinstance(block, VariableBlock)
                    return self._exec_variable(block)
                case BlockType.BASH:
                    assert isinstance(block, BashBlock)
                    return await self._exec_bash(block)
                case BlockType.COMMAND:
                    assert isinstance(block, CommandBlock)
                    return await self._exec_command(block)
                case BlockType.REFRESH:
                    assert isinstance(block, RefreshBlock)
                    return await self._exec_refresh(block)
                case BlockType.SPAWN:
                    assert isinstance(block, SpawnBlock)
                    return await self._exec_spawn(block)
                case BlockType.WAIT:
                    assert isinstance(block, WaitBlock)
                    return await self._exec_wait(block)
                case BlockType.EXIT:
                    assert isinstance(block, ExitBlock)
                    return self._exec_exit(block)
                case BlockType.INPUT:
                    assert isinstance(block, InputBlock)
                    return await self._exec_input(block)
                case _:
                    return BlockResult.fail(f"Unknown block type: {block.type}")

    async def _exec_prompt(self, block: PromptBlock) -> BlockResult:
        """Execute a prompt block: resolve template, send to session, collect response."""
        with _tracer.start_as_current_span("flowchart.exec_prompt", attributes={"block.name": block.name}):
            prompt_text = evaluate_template(block.prompt, self._variables)

            if block.session != "default":
                self._protocol.log(
                    f"WARNING: Multi-session not yet supported, "
                    f"block \"{block.name}\" requests session \"{block.session}\" "
                    f"— using main session"
                )

            # Append output schema instructions if specified
            if block.output_schema:
                schema_str = json.dumps(block.output_schema, indent=2)
                prompt_text += (
                    f"\n\nRespond with JSON matching this schema:\n"
                    f"```json\n{schema_str}\n```"
                )

            start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    self._session.query(
                        prompt_text, block_id=block.id, block_name=block.name
                    ),
                    timeout=block.timeout_seconds,
                )
            except asyncio.TimeoutError:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                # Kill the Claude subprocess so the session doesn't keep running.
                await self._session.stop()
                self._protocol.emit_block_timeout(
                    block.id,
                    block.name,
                    block.type,
                    elapsed_ms,
                    block.timeout_seconds,
                )
                return BlockResult.fail(
                    f"Block timed out after {elapsed_ms}ms "
                    f"(limit: {block.timeout_seconds}s)"
                )

            # Save raw output to variable if requested
            if block.output_variable and result.response_text:
                self._variables[block.output_variable] = result.response_text

            # Parse structured output
            if block.output_schema and result.response_text:
                parsed = parse_json_from_response(result.response_text)
                if parsed:
                    self._variables.update(parsed)

            return BlockResult.ok(output=result.response_text)

    def _exec_branch(self, block: BranchBlock) -> BlockResult:
        """Evaluate a branch condition against variables.

        Supports:
        - Simple variable truthiness: ``fullyImplemented``
        - Negation: ``!hasErrors``
        - Comparison operators: ``exitCode == 0``, ``i < 3``, ``status != "done"``
        - Template refs in conditions: ``i < {{max}}``
        """
        # Resolve any {{var}} / $N templates in the condition text first
        condition = evaluate_template(block.condition, self._variables)
        truthy = _evaluate_condition(condition, self._variables)
        self._protocol.log(
            f"Branch \"{block.name}\": {block.condition} -> {truthy}"
        )
        return BlockResult.ok(branch_taken=truthy)

    def _exec_variable(self, block: VariableBlock) -> BlockResult:
        """Set a variable with type coercion."""
        raw = evaluate_template(block.variable_value, self._variables)

        try:
            match block.variable_type:
                case VariableType.STRING:
                    value: Any = raw
                case VariableType.NUMBER:
                    value = float(raw)
                case VariableType.BOOLEAN:
                    value = raw.lower() in ("true", "1", "yes")
                case VariableType.JSON:
                    value = json.loads(raw)
                case _:
                    value = raw
        except (ValueError, json.JSONDecodeError) as e:
            return BlockResult.fail(
                f"Variable '{block.variable_name}' type coercion failed: {e}"
            )

        self._variables[block.variable_name] = value
        return BlockResult.ok()

    async def _exec_bash(self, block: BashBlock) -> BlockResult:
        """Execute a shell command."""
        cmd = evaluate_template(block.command, self._variables)
        with _tracer.start_as_current_span(
            "flowchart.exec_bash",
            attributes={"block.name": block.name, "bash.command": cmd[:200]},
        ) as span:
            cwd = block.working_directory or None

            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=PIPE, stderr=PIPE, cwd=cwd
                )
                stdout, stderr = await proc.communicate()
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                return BlockResult.fail(f"Failed to run command: {e}")

            exit_code = proc.returncode or 0
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            span.set_attribute("bash.exit_code", exit_code)

            # Store exit code variable
            if block.exit_code_variable:
                self._variables[block.exit_code_variable] = exit_code

            # Check exit code
            if exit_code != 0 and not block.continue_on_error:
                return BlockResult.fail(
                    f"Exit code {exit_code}: {stderr_str or stdout_str}"
                )

            # Capture output
            if block.capture_output and block.output_variable:
                output = stdout_str.strip()
                try:
                    match block.output_type:
                        case VariableType.NUMBER:
                            self._variables[block.output_variable] = float(output)
                        case VariableType.BOOLEAN:
                            self._variables[block.output_variable] = output.lower() in (
                                "true", "1", "yes",
                            )
                        case VariableType.JSON:
                            self._variables[block.output_variable] = json.loads(output)
                        case _:
                            self._variables[block.output_variable] = output
                except (ValueError, json.JSONDecodeError):
                    self._variables[block.output_variable] = output

            return BlockResult.ok(output=stdout_str)

    async def _exec_refresh(self, block: RefreshBlock) -> BlockResult:
        """Clear conversation history by restarting the session."""
        _tracer.start_span("flowchart.exec_refresh", attributes={"block.name": block.name}).end()
        await self._session.clear()
        self._protocol.log("Session cleared (conversation history reset)")
        return BlockResult.ok()

    async def _exec_command(self, block: CommandBlock) -> BlockResult:
        """Execute a sub-command (composition).

        Resolves the command by name, creates a child walker with scoped
        variables, and runs it. Recursion is tracked via the call stack.
        """
        depth = len(self._call_stack)
        command_name = evaluate_template(block.command_name, self._variables)
        with _tracer.start_as_current_span(
            "flowchart.exec_command",
            attributes={"command.name": command_name, "command.depth": depth},
        ) as span:
            if depth >= self._max_depth:
                return BlockResult.fail(
                    f"Max recursion depth ({self._max_depth}) exceeded. "
                    f"Call stack: {' -> '.join(self._call_stack)}"
                )

            # Check for direct recursion (same command already in stack)
            if command_name in self._call_stack:
                self._protocol.log(
                    f"WARNING: Recursive call to '{command_name}' "
                    f"(depth {depth}). Stack: {' -> '.join(self._call_stack)}"
                )

            # Resolve the command
            try:
                cmd = resolve_command(command_name, search_paths=self._search_paths)
            except CommandNotFoundError as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                return BlockResult.fail(str(e))

            # Build child variables
            child_vars: dict[str, Any] = {}
            if block.inherit_variables:
                child_vars = dict(self._variables)

            # Parse arguments and map to $1, $2, etc.
            if block.arguments:
                arg_text = evaluate_template(block.arguments, self._variables)
                try:
                    parts = shlex.split(arg_text)
                except ValueError:
                    parts = arg_text.split()
                for i, part in enumerate(parts, 1):
                    child_vars[f"${i}"] = part

            self._protocol.log(
                f"Entering sub-command '{command_name}' (depth {depth + 1})"
            )

            # Create child walker sharing the same session and protocol
            child_walker = GraphWalker(
                cmd.flowchart,
                self._session,
                child_vars,
                self._protocol,
                max_blocks=self._max_blocks,
                call_stack=[*self._call_stack, command_name],
                max_depth=self._max_depth,
                search_paths=self._search_paths,
                session_factory=self._session_factory,
            )

            child_result = await child_walker.run()

            self._protocol.log(
                f"Sub-command '{command_name}' finished: {child_result.status}"
            )

            if child_result.status != "completed":
                # Include the child's error for better diagnostics
                child_errors = [
                    e.result.error for e in child_result.log if e.result.error
                ]
                detail = child_errors[-1] if child_errors else child_result.status
                span.set_status(trace.StatusCode.ERROR, detail)
                return BlockResult.fail(
                    f"Sub-command '{block.command_name}' failed: {detail}"
                )

            # Merge output variables back into parent scope
            if block.merge_output:
                # Don't merge positional args ($1, $2, etc.) back
                for k, v in child_result.variables.items():
                    if not k.startswith("$"):
                        self._variables[k] = v

            return BlockResult.ok(output=json.dumps(child_result.variables))

    async def _exec_spawn(self, block: SpawnBlock) -> BlockResult:
        """Spawn a named agent sub-session running a command asynchronously."""
        agent_name = evaluate_template(block.agent_name, self._variables)

        if agent_name in self._spawned_tasks:
            return BlockResult.fail(
                f"Agent '{agent_name}' is already spawned. "
                f"Wait for it before spawning again."
            )

        command_name = evaluate_template(block.command_name, self._variables)
        resolved_model = (
            evaluate_template(block.model, self._variables)
            if block.model
            else block.model
        )

        try:
            cmd = resolve_command(command_name, search_paths=self._search_paths)
        except CommandNotFoundError as e:
            return BlockResult.fail(str(e))

        child_vars: dict[str, Any] = {}
        if block.inherit_variables:
            child_vars = dict(self._variables)

        if block.arguments:
            arg_text = evaluate_template(block.arguments, self._variables)
            try:
                parts = shlex.split(arg_text)
            except ValueError:
                parts = arg_text.split()
            for i, part in enumerate(parts, 1):
                child_vars[f"${i}"] = part

        # Optional per-spawn working directory: template it (so it can use
        # {{var}}/$N), create it, and hand it to the child so parallel siblings
        # can be isolated on disk.  Unset -> None -> child inherits parent cwd.
        spawn_cwd: str | None = None
        if block.cwd:
            spawn_cwd = evaluate_template(block.cwd, self._variables)
            os.makedirs(spawn_cwd, exist_ok=True)

        if block.backend and self._session_factory:
            # NOTE: the backend path builds via SessionCreator(name, model),
            # which has no cwd parameter, so block.cwd is not honored here yet.
            # Follow-up: thread cwd through SessionFactory.create / SessionCreator.
            child_session = self._session_factory.create(
                block.backend, agent_name, resolved_model
            )
            if spawn_cwd is not None:
                self._protocol.log(
                    f"Note: per-spawn cwd is not applied to backend-spawned "
                    f"agent '{agent_name}' (backend factory has no cwd param)"
                )
            self._protocol.log(
                f"Spawning agent '{agent_name}' with backend '{block.backend}'"
                f"{f' model {resolved_model!r}' if resolved_model else ''} "
                f"running command '{command_name}'"
            )
        elif resolved_model:
            child_session = self._session.with_model(resolved_model).clone(
                agent_name, cwd=spawn_cwd
            )
            self._protocol.log(
                f"Spawning agent '{agent_name}' with model '{resolved_model}' "
                f"running command '{command_name}'"
            )
        else:
            child_session = self._session.clone(agent_name, cwd=spawn_cwd)
            self._protocol.log(
                f"Spawning agent '{agent_name}' running command '{command_name}'"
            )

        await child_session.start()

        child_walker = GraphWalker(
            cmd.flowchart,
            child_session,
            child_vars,
            self._protocol,
            max_blocks=self._max_blocks,
            call_stack=[*self._call_stack, f"spawn:{block.command_name}"],
            max_depth=self._max_depth,
            search_paths=self._search_paths,
            session_factory=self._session_factory,
        )

        task = asyncio.create_task(child_walker.run())
        self._spawned_tasks[agent_name] = task
        self._spawned_sessions[agent_name] = child_session

        return BlockResult.ok(output=f"Spawned agent '{agent_name}'")

    async def _exec_wait(self, block: WaitBlock) -> BlockResult:
        """Wait for spawned agent sessions to complete."""
        # Template-evaluate each wait_for entry so a wait can name a child whose
        # agent_name was itself templated (e.g. wait_for=["w-{{i}}"] joining a
        # spawn of "w-{{i}}"). Mirrors _exec_spawn's handling of agent_name. The
        # empty (join-all) case resolves to the live task keys, which are already
        # resolved. De-dup while preserving order (a template can collapse to
        # repeats, and a repeated name would miss on the second lookup).
        if block.wait_for:
            wait_for: list[str] = []
            for name in block.wait_for:
                resolved = evaluate_template(name, self._variables)
                if resolved not in wait_for:
                    wait_for.append(resolved)
        else:
            wait_for = list(self._spawned_tasks.keys())

        if not wait_for:
            self._protocol.log("Wait block: no spawned agents to wait for")
            return BlockResult.ok()

        errors: list[str] = []
        for agent_name in wait_for:
            task = self._spawned_tasks.get(agent_name)
            if not task:
                errors.append(f"No spawned agent named '{agent_name}'")
                continue

            try:
                if block.timeout_seconds:
                    exec_result = await asyncio.wait_for(
                        task, timeout=block.timeout_seconds
                    )
                else:
                    exec_result = await task
            except asyncio.TimeoutError:
                errors.append(
                    f"Agent '{agent_name}' timed out after {block.timeout_seconds}s"
                )
                task.cancel()
                continue
            except Exception as e:
                errors.append(f"Agent '{agent_name}' failed: {e}")
                continue

            self._spawned_results[agent_name] = exec_result
            self._protocol.log(
                f"Agent '{agent_name}' completed: {exec_result.status}"
            )

            # Roll this child's cumulative cost up into the parent session's
            # total_cost. Unlike cost_variable (opt-in, observability only),
            # this always aggregates nested spawn costs, so a wrapper that
            # spawns a doer (no cost_variable) still reports the doer's cost.
            # Each child is popped from _spawned_tasks below, so it's rolled up
            # exactly once; the recursion (each wait level rolls up its own
            # children) makes a top-level session's cost include deeply-nested
            # descendants. No double-count.
            self._session.add_cost(exec_result.cost_usd)

            # Store per-child metrics (exit code, cost, duration, token
            # breakdown) into the parent variables named on the matching
            # spawn block.  Each variable NAME is template-evaluated (like
            # agent_name), so a fan-out can target a unique variable per child,
            # e.g. cost_variable: "cost_{{i}}" -> cost_1, cost_2, ...  A literal
            # name (no {{...}}) passes through unchanged.
            for b in self._flowchart.blocks.values():
                if not isinstance(b, SpawnBlock):
                    continue
                if evaluate_template(b.agent_name, self._variables) != agent_name:
                    continue
                for var_name, value in (
                    (b.exit_code_variable, exec_result.exit_code),
                    (b.cost_variable, exec_result.cost_usd),
                    (b.duration_variable, exec_result.duration_ms),
                    (b.input_tokens_variable, exec_result.input_tokens),
                    (b.output_tokens_variable, exec_result.output_tokens),
                    (b.cache_creation_tokens_variable, exec_result.cache_creation_tokens),
                    (b.cache_read_tokens_variable, exec_result.cache_read_tokens),
                ):
                    if var_name:
                        key = evaluate_template(var_name, self._variables)
                        self._variables[key] = value

            # Clean up the spawned session
            spawned_session = self._spawned_sessions.get(agent_name)
            if spawned_session and spawned_session is not self._session:
                await spawned_session.stop()
            self._spawned_tasks.pop(agent_name, None)
            self._spawned_sessions.pop(agent_name, None)

        if errors:
            return BlockResult.fail("; ".join(errors))

        return BlockResult.ok()

    def _exec_exit(self, block: ExitBlock) -> BlockResult:
        """Handle an explicit exit with a given exit code."""
        message = (
            evaluate_template(block.exit_message, self._variables)
            if block.exit_message
            else ""
        )
        self._protocol.log(
            f"Exit block '{block.name}': code={block.exit_code}, message={message}"
        )
        return BlockResult.exit(code=block.exit_code, message=message)

    async def _exec_input(self, block: InputBlock) -> BlockResult:
        """Pause and wait for user input, send to agent, optionally capture response."""
        self._protocol.log(f"Input block '{block.name}': waiting for user input")
        self._protocol.emit_system(
            "input_request",
            {"block_id": block.id, "block_name": block.name},
        )

        # Wait for this block's input_response, bounded by timeout_seconds so a
        # missing response fails the block instead of hanging forever (there was
        # no bound here before).  With delivery wired up during takeover, a
        # non-matching message must not be re-queued into the same inbox we are
        # draining or the loop busy-spins, so such messages are buffered and
        # re-pushed once, after the wait.
        wait_start = time.monotonic()
        buffered: list[dict] = []
        user_text = ""
        try:
            while True:
                remaining = block.timeout_seconds - (time.monotonic() - wait_start)
                try:
                    if remaining <= 0:
                        raise TimeoutError
                    msg = await asyncio.wait_for(
                        self._protocol.read_message(), remaining
                    )
                except TimeoutError:
                    elapsed_ms = int((time.monotonic() - wait_start) * 1000)
                    self._protocol.emit_block_timeout(
                        block.id,
                        block.name,
                        block.type,
                        elapsed_ms,
                        block.timeout_seconds,
                    )
                    return BlockResult.fail(
                        f"Input block '{block.name}': no input received "
                        f"within {block.timeout_seconds}s"
                    )
                if (
                    msg.get("type") == "input_response"
                    and msg.get("block_id") == block.id
                ):
                    user_text = msg.get("content", "")
                    break
                buffered.append(msg)
        finally:
            for pending in buffered:
                self._protocol.push_message(pending)

        if not user_text:
            self._protocol.log(f"Input block '{block.name}': received empty input")
            return BlockResult.ok(output="")

        start = time.monotonic()
        try:
            result = await self._session.query(
                user_text, block_id=block.id, block_name=block.name
            )
        except ReadTimeoutError:
            # The inner CLI went silent after the input was delivered.  Report
            # a failed block instead of letting the flowchart hang forever.
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._protocol.emit_block_timeout(
                block.id, block.name, block.type, elapsed_ms, block.timeout_seconds
            )
            return BlockResult.fail(
                f"Input block '{block.name}': agent stopped responding "
                f"after {elapsed_ms}ms (CLI idle timeout)"
            )
        except ReadInterruptedError:
            return BlockResult.fail(
                f"Input block '{block.name}': interrupted while "
                f"waiting for the agent"
            )

        if block.output_variable and result.response_text:
            self._variables[block.output_variable] = result.response_text

        return BlockResult.ok(output=result.response_text)

    async def _cleanup_spawned(self) -> None:
        """Cancel and clean up any remaining spawned tasks and sessions."""
        for agent_name, task in list(self._spawned_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        for agent_name, session in list(self._spawned_sessions.items()):
            if session is not self._session:
                await session.stop()
        self._spawned_tasks.clear()
        self._spawned_sessions.clear()

    def _find_start_block(self) -> BlockBase | None:
        """Find the start block in the flowchart."""
        for block in self._flowchart.blocks.values():
            if block.type == BlockType.START:
                return block
        return None

    def _next_block(self, current: BlockBase, result: BlockResult) -> BlockBase | None:
        """Find the next block to execute based on connections."""
        if current.type in (BlockType.END, BlockType.EXIT):
            return None

        outgoing = [
            c for c in self._flowchart.connections if c.source_id == current.id
        ]

        if current.type == BlockType.BRANCH:
            for conn in outgoing:
                if conn.is_true_path == result.branch_taken:
                    return self._flowchart.blocks.get(conn.target_id)
            return None

        # Non-branch: take the first (only) connection
        if outgoing:
            return self._flowchart.blocks.get(outgoing[0].target_id)
        return None
