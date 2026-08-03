"""Integration tests for Spawn/Wait blocks.

Tests the full spawn/wait control flow using real command resolution
and real asyncio task spawning. No mocks — sub-commands are saved to
a temp directory and resolved via search_paths.

Tests marked @pytest.mark.slow use a real claude CLI session and cost
tokens. Run with: pytest -m slow
"""

import json
import shutil
import pytest
from pathlib import Path

from flowcoder_flowchart import (
    BashBlock,
    Command,
    Connection,
    EndBlock,
    Flowchart,
    PromptBlock,
    SpawnBlock,
    StartBlock,
    VariableBlock,
    VariableType,
    WaitBlock,
    save_command,
)
from flowcoder_engine.walker import GraphWalker
from flowcoder_engine.session import Session
from flowcoder_engine.cli import build_inner_claude_cmd, parse_args

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "flowcoder-engine" / "tests"))
from conftest import MockSession, MockProtocol


@pytest.fixture
def cmd_dir(tmp_path):
    """Temp directory for sub-command JSON files."""
    return tmp_path


def _save_cmd(cmd_dir: Path, command: Command) -> None:
    """Save a command to the temp directory for resolve_command to find."""
    save_command(command, cmd_dir / f"{command.name}.json")


def _make_bash_command(name: str, script: str, output_var: str = "result") -> Command:
    """Create a command with a single bash block."""
    return Command(
        name=name,
        flowchart=Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "b": BashBlock(
                    id="b", name="Run",
                    command=script,
                    capture_output=True,
                    output_variable=output_var,
                ),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="b"),
                Connection(source_id="b", target_id="e"),
            ],
        ),
    )


class TestSpawnWaitBash:
    """Test spawn/wait with bash-only sub-commands (no LLM cost).

    The session is passed to GraphWalker but never queried because
    bash blocks don't use it. This tests the full spawn/wait control
    flow: command resolution, child walker creation, asyncio task
    management, and result collection.
    """

    @pytest.mark.asyncio
    async def test_spawn_wait_basic(self, cmd_dir):
        """Spawn a bash sub-command and wait for it to complete."""
        sub_cmd = _make_bash_command("echo-worker", "echo hello")
        _save_cmd(cmd_dir, sub_cmd)

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn Worker",
                    agent_name="worker",
                    command_name="echo-worker",
                ),
                "w": WaitBlock(id="w", name="Wait", wait_for=["worker"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        assert result.status == "completed"
        block_types = [e.block_type for e in result.log]
        assert "spawn" in block_types
        assert "wait" in block_types

    @pytest.mark.asyncio
    async def test_spawn_wait_captures_exit_code(self, cmd_dir):
        """Spawn block's exit_code_variable captures child exit code."""
        sub_cmd = _make_bash_command("success-worker", "echo done")
        _save_cmd(cmd_dir, sub_cmd)

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn",
                    agent_name="worker",
                    command_name="success-worker",
                    exit_code_variable="worker_exit",
                ),
                "w": WaitBlock(id="w", name="Wait", wait_for=["worker"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        assert result.status == "completed"
        assert result.variables.get("worker_exit") == 0

    @pytest.mark.asyncio
    async def test_spawn_wait_multiple_agents(self, cmd_dir):
        """Spawn two agents and wait for both."""
        _save_cmd(cmd_dir, _make_bash_command("worker-a", "echo alpha"))
        _save_cmd(cmd_dir, _make_bash_command("worker-b", "echo bravo"))

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp1": SpawnBlock(
                    id="sp1", name="Spawn A",
                    agent_name="alpha",
                    command_name="worker-a",
                ),
                "sp2": SpawnBlock(
                    id="sp2", name="Spawn B",
                    agent_name="bravo",
                    command_name="worker-b",
                ),
                "w": WaitBlock(id="w", name="Wait All"),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp1"),
                Connection(source_id="sp1", target_id="sp2"),
                Connection(source_id="sp2", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_spawn_inherits_variables(self, cmd_dir):
        """Spawn with inherit_variables passes parent variables to child."""
        # Sub-command uses $1 from inherited variable
        sub_cmd = Command(
            name="greeter",
            flowchart=Flowchart(
                blocks={
                    "s": StartBlock(id="s", name="Start"),
                    "b": BashBlock(
                        id="b", name="Greet",
                        command="echo hello {{name}}",
                        capture_output=True,
                        output_variable="greeting",
                    ),
                    "e": EndBlock(id="e", name="End"),
                },
                connections=[
                    Connection(source_id="s", target_id="b"),
                    Connection(source_id="b", target_id="e"),
                ],
            ),
        )
        _save_cmd(cmd_dir, sub_cmd)

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "v": VariableBlock(
                    id="v", name="Set Name",
                    variable_name="name", variable_value="world",
                ),
                "sp": SpawnBlock(
                    id="sp", name="Spawn",
                    agent_name="greeter",
                    command_name="greeter",
                    inherit_variables=True,
                ),
                "w": WaitBlock(id="w", name="Wait", wait_for=["greeter"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="v"),
                Connection(source_id="v", target_id="sp"),
                Connection(source_id="sp", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_spawn_with_arguments(self, cmd_dir):
        """Spawn block passes arguments to child command."""
        sub_cmd = Command(
            name="arg-worker",
            flowchart=Flowchart(
                blocks={
                    "s": StartBlock(id="s", name="Start"),
                    "b": BashBlock(
                        id="b", name="Use Arg",
                        command="echo $1",
                        capture_output=True,
                        output_variable="result",
                    ),
                    "e": EndBlock(id="e", name="End"),
                },
                connections=[
                    Connection(source_id="s", target_id="b"),
                    Connection(source_id="b", target_id="e"),
                ],
            ),
        )
        _save_cmd(cmd_dir, sub_cmd)

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn",
                    agent_name="worker",
                    command_name="arg-worker",
                    arguments="test-value",
                ),
                "w": WaitBlock(id="w", name="Wait", wait_for=["worker"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_spawn_duplicate_agent_fails(self, cmd_dir):
        """Spawning the same agent name twice fails."""
        _save_cmd(cmd_dir, _make_bash_command("slow-worker", "sleep 0.1 && echo done"))

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp1": SpawnBlock(
                    id="sp1", name="Spawn 1",
                    agent_name="worker",
                    command_name="slow-worker",
                ),
                "sp2": SpawnBlock(
                    id="sp2", name="Spawn 2",
                    agent_name="worker",
                    command_name="slow-worker",
                ),
                "w": WaitBlock(id="w", name="Wait", wait_for=["worker"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp1"),
                Connection(source_id="sp1", target_id="sp2"),
                Connection(source_id="sp2", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        # Second spawn should fail because "worker" is already spawned
        assert result.status == "halted"
        error_entries = [e for e in result.log if e.result.error]
        assert any("already spawned" in e.result.error for e in error_entries)

    @pytest.mark.asyncio
    async def test_wait_unknown_agent_fails(self, cmd_dir):
        """Waiting for a non-existent agent fails."""
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "w": WaitBlock(id="w", name="Wait", wait_for=["ghost"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        assert result.status == "halted"
        error_entries = [e for e in result.log if e.result.error]
        assert any("ghost" in e.result.error for e in error_entries)

    @pytest.mark.asyncio
    async def test_spawn_nonexistent_command_fails(self, cmd_dir):
        """Spawning a command that doesn't exist fails."""
        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn",
                    agent_name="worker",
                    command_name="nonexistent-command",
                ),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        assert result.status == "halted"
        error_entries = [e for e in result.log if e.result.error]
        assert any("not found" in e.result.error.lower() for e in error_entries)

    @pytest.mark.asyncio
    async def test_wait_empty_waits_for_all(self, cmd_dir):
        """WaitBlock with empty wait_for waits for all spawned agents."""
        _save_cmd(cmd_dir, _make_bash_command("fast-worker", "echo fast"))

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp1": SpawnBlock(
                    id="sp1", name="Spawn A",
                    agent_name="a",
                    command_name="fast-worker",
                ),
                "sp2": SpawnBlock(
                    id="sp2", name="Spawn B",
                    agent_name="b",
                    command_name="fast-worker",
                ),
                "w": WaitBlock(id="w", name="Wait All"),  # empty wait_for = all
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp1"),
                Connection(source_id="sp1", target_id="sp2"),
                Connection(source_id="sp2", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(),
            search_paths=[str(cmd_dir)],
        )
        result = await walker.run()

        assert result.status == "completed"


class TestSpawnWaitClaude:
    """Integration tests using a real claude CLI session.

    These tests spawn a real claude subprocess, cost tokens, and are
    slower. Run with: pytest -m slow

    Requires: claude CLI installed at ~/.local/bin/claude
    """

    CLAUDE_PATH = str(Path.home() / ".local" / "bin" / "claude")

    @classmethod
    def build_claude_cmd(cls, *extra: str) -> list[str]:
        """Build an inner-CLI command the same way the engine does.

        Session takes claude_cmd verbatim — session.py copies the list and
        subprocess.py execs it, with nothing prepended — so a command that
        omits the protocol flags launches an interactive CLI that never
        writes stream-json.  The session then receives nothing at all and
        the run halts once the read bound fires.

        Delegating to the engine's own builder rather than restating the
        flags keeps these tests from drifting: if the engine changes how it
        launches the CLI, they follow automatically.
        """
        return build_inner_claude_cmd(parse_args(list(extra)), cls.CLAUDE_PATH)

    @pytest.fixture
    def has_claude(self):
        """Skip if claude CLI is not installed."""
        if not Path(self.CLAUDE_PATH).exists():
            pytest.skip("claude CLI not found")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_spawn_prompt_wait(self, cmd_dir, has_claude):
        """Full integration: spawn a command with a prompt block, wait, check output."""
        # Sub-command asks Claude a simple question
        sub_cmd = Command(
            name="hello-agent",
            flowchart=Flowchart(
                blocks={
                    "s": StartBlock(id="s", name="Start"),
                    "p": PromptBlock(
                        id="p", name="Say Hello",
                        prompt=(
                            "Respond with exactly one word: hello\n"
                            "Do not include any other text, punctuation, or formatting."
                        ),
                        output_variable="response",
                    ),
                    "e": EndBlock(id="e", name="End"),
                },
                connections=[
                    Connection(source_id="s", target_id="p"),
                    Connection(source_id="p", target_id="e"),
                ],
            ),
        )
        _save_cmd(cmd_dir, sub_cmd)

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn Hello",
                    agent_name="greeter",
                    command_name="hello-agent",
                    exit_code_variable="greeter_exit",
                ),
                "w": WaitBlock(id="w", name="Wait", wait_for=["greeter"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        # Create a real session
        session = Session(
            name="test",
            claude_cmd=self.build_claude_cmd(
                "--model", "claude-haiku-4-5-20251001",
                "--max-turns", "1",
            ),
            protocol=MockProtocol(),
        )

        try:
            await session.start()

            walker = GraphWalker(
                fc, session, {}, MockProtocol(),
                search_paths=[str(cmd_dir)],
            )
            result = await walker.run()

            assert result.status == "completed"
            assert result.variables.get("greeter_exit") == 0
        finally:
            await session.stop()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_spawn_with_model_override(self, cmd_dir, has_claude):
        """Spawn with model creates a dedicated session using that model."""
        sub_cmd = Command(
            name="model-agent",
            flowchart=Flowchart(
                blocks={
                    "s": StartBlock(id="s", name="Start"),
                    "p": PromptBlock(
                        id="p", name="Say Hi",
                        prompt=(
                            "Respond with exactly one word: hi\n"
                            "Do not include any other text, punctuation, or formatting."
                        ),
                        output_variable="response",
                    ),
                    "e": EndBlock(id="e", name="End"),
                },
                connections=[
                    Connection(source_id="s", target_id="p"),
                    Connection(source_id="p", target_id="e"),
                ],
            ),
        )
        _save_cmd(cmd_dir, sub_cmd)

        fc = Flowchart(
            blocks={
                "s": StartBlock(id="s", name="Start"),
                "sp": SpawnBlock(
                    id="sp", name="Spawn Model",
                    agent_name="model-worker",
                    command_name="model-agent",
                    exit_code_variable="worker_exit",
                    model="claude-haiku-4-5-20251001",
                ),
                "w": WaitBlock(id="w", name="Wait", wait_for=["model-worker"]),
                "e": EndBlock(id="e", name="End"),
            },
            connections=[
                Connection(source_id="s", target_id="sp"),
                Connection(source_id="sp", target_id="w"),
                Connection(source_id="w", target_id="e"),
            ],
        )

        # Parent session uses haiku — spawn also uses haiku (via model override)
        # The key is that a *separate* Session is created for the spawned agent
        session = Session(
            name="test-parent",
            claude_cmd=self.build_claude_cmd(
                "--model", "claude-haiku-4-5-20251001",
                "--max-turns", "1",
            ),
            protocol=MockProtocol(),
        )

        try:
            await session.start()

            walker = GraphWalker(
                fc, session, {}, MockProtocol(),
                search_paths=[str(cmd_dir)],
            )
            result = await walker.run()

            assert result.status == "completed"
            assert result.variables.get("worker_exit") == 0
        finally:
            await session.stop()
