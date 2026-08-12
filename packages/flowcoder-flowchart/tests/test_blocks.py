"""Tests for block types and discriminated union."""

import pytest
from flowcoder_flowchart import (
    BashBlock,
    Block,
    BlockType,
    BranchBlock,
    CommandBlock,
    EndBlock,
    ExitBlock,
    InputBlock,
    Position,
    PromptBlock,
    RefreshBlock,
    SpawnBlock,
    StartBlock,
    VariableBlock,
    VariableType,
    WaitBlock,
)
from pydantic import TypeAdapter

BlockAdapter = TypeAdapter(Block)


class TestBlockCreation:
    def test_start_block(self):
        b = StartBlock(name="Start")
        assert b.type == BlockType.START
        assert b.name == "Start"
        assert b.session == "default"
        assert b.id  # auto-generated

    def test_end_block(self):
        b = EndBlock(name="End")
        assert b.type == BlockType.END

    def test_prompt_block(self):
        b = PromptBlock(name="Ask", prompt="Hello $1")
        assert b.type == BlockType.PROMPT
        assert b.prompt == "Hello $1"
        assert b.output_schema is None

    def test_prompt_block_with_schema(self):
        schema = {"valid": "boolean", "reason": "string"}
        b = PromptBlock(name="Check", prompt="Is this valid?", output_schema=schema)
        assert b.output_schema == schema

    def test_branch_block(self):
        b = BranchBlock(name="Check", condition="is_valid")
        assert b.type == BlockType.BRANCH
        assert b.condition == "is_valid"

    def test_variable_block(self):
        b = VariableBlock(
            name="Set X",
            variable_name="x",
            variable_value="42",
            variable_type=VariableType.NUMBER,
        )
        assert b.type == BlockType.VARIABLE
        assert b.variable_type == VariableType.NUMBER

    def test_bash_block(self):
        b = BashBlock(
            name="Run",
            command="echo hello",
            output_variable="result",
            exit_code_variable="rc",
            working_directory="/tmp",
        )
        assert b.type == BlockType.BASH
        assert b.capture_output is True
        assert b.continue_on_error is False
        assert b.working_directory == "/tmp"
        assert b.exit_code_variable == "rc"

    def test_command_block(self):
        b = CommandBlock(
            name="Call Sub",
            command_name="sub-flow",
            arguments="arg1 arg2",
            inherit_variables=True,
            merge_output=True,
        )
        assert b.type == BlockType.COMMAND
        assert b.command_name == "sub-flow"

    def test_refresh_block(self):
        b = RefreshBlock(name="Reset", target_session="reviewer")
        assert b.type == BlockType.REFRESH
        assert b.target_session == "reviewer"

    def test_refresh_block_default_session(self):
        b = RefreshBlock(name="Reset")
        assert b.target_session is None

    def test_position(self):
        b = StartBlock(name="Start", position=Position(x=100, y=200))
        assert b.position.x == 100
        assert b.position.y == 200

    def test_custom_session(self):
        b = PromptBlock(name="Ask", prompt="hi", session="reviewer")
        assert b.session == "reviewer"

    def test_custom_id(self):
        b = StartBlock(id="my-id", name="Start")
        assert b.id == "my-id"


class TestBlockDiscriminatedUnion:
    def test_deserialize_start(self):
        b = BlockAdapter.validate_python({"type": "start", "name": "S"})
        assert isinstance(b, StartBlock)

    def test_deserialize_end(self):
        b = BlockAdapter.validate_python({"type": "end", "name": "E"})
        assert isinstance(b, EndBlock)

    def test_deserialize_prompt(self):
        b = BlockAdapter.validate_python(
            {"type": "prompt", "name": "P", "prompt": "Hello"}
        )
        assert isinstance(b, PromptBlock)
        assert b.prompt == "Hello"

    def test_deserialize_branch(self):
        b = BlockAdapter.validate_python(
            {"type": "branch", "name": "B", "condition": "x"}
        )
        assert isinstance(b, BranchBlock)

    def test_deserialize_variable(self):
        b = BlockAdapter.validate_python(
            {"type": "variable", "name": "V", "variable_name": "x"}
        )
        assert isinstance(b, VariableBlock)

    def test_deserialize_bash(self):
        b = BlockAdapter.validate_python(
            {"type": "bash", "name": "B", "command": "ls"}
        )
        assert isinstance(b, BashBlock)

    def test_deserialize_command(self):
        b = BlockAdapter.validate_python(
            {"type": "command", "name": "C", "command_name": "sub"}
        )
        assert isinstance(b, CommandBlock)

    def test_deserialize_refresh(self):
        b = BlockAdapter.validate_python({"type": "refresh", "name": "R"})
        assert isinstance(b, RefreshBlock)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            BlockAdapter.validate_python({"type": "invalid", "name": "X"})

    def test_roundtrip(self):
        original = PromptBlock(
            id="p1",
            name="Ask",
            prompt="Hello $1",
            session="reviewer",
            output_schema={"answer": "string"},
        )
        data = original.model_dump()
        restored = BlockAdapter.validate_python(data)
        assert isinstance(restored, PromptBlock)
        assert restored.id == "p1"
        assert restored.prompt == "Hello $1"
        assert restored.session == "reviewer"
        assert restored.output_schema == {"answer": "string"}

    def test_roundtrip_json(self):
        original = BashBlock(
            id="b1",
            name="Run",
            command="echo hi",
            working_directory="/tmp",
            exit_code_variable="rc",
        )
        json_str = original.model_dump_json()
        restored = BlockAdapter.validate_json(json_str)
        assert isinstance(restored, BashBlock)
        assert restored.working_directory == "/tmp"
        assert restored.exit_code_variable == "rc"


class TestSpawnBlock:
    def test_creation(self):
        b = SpawnBlock(
            name="Spawn Worker",
            agent_name="worker-1",
            command_name="sub-flow",
            arguments="--fast",
            model="haiku",
        )
        assert b.type == BlockType.SPAWN
        assert b.agent_name == "worker-1"
        assert b.command_name == "sub-flow"
        assert b.arguments == "--fast"
        assert b.model == "haiku"
        assert b.inherit_variables is False
        assert b.exit_code_variable is None
        assert b.cost_variable is None
        assert b.duration_variable is None
        assert b.input_tokens_variable is None
        assert b.output_tokens_variable is None
        assert b.cache_creation_tokens_variable is None
        assert b.cache_read_tokens_variable is None
        assert b.config_file is None
        assert b.cwd is None

    def test_defaults(self):
        b = SpawnBlock()
        assert b.agent_name == ""
        assert b.command_name == ""
        assert b.arguments == ""
        assert b.model is None

    def test_deserialize(self):
        b = BlockAdapter.validate_python({
            "type": "spawn",
            "name": "S",
            "command_name": "sub",
            "agent_name": "a1",
        })
        assert isinstance(b, SpawnBlock)
        assert b.command_name == "sub"
        assert b.agent_name == "a1"

    def test_roundtrip(self):
        original = SpawnBlock(
            id="sp1",
            name="Spawn",
            agent_name="worker",
            command_name="build",
            model="opus",
            inherit_variables=True,
            exit_code_variable="rc",
            cost_variable="cost",
        )
        data = original.model_dump()
        restored = BlockAdapter.validate_python(data)
        assert isinstance(restored, SpawnBlock)
        assert restored.agent_name == "worker"
        assert restored.model == "opus"
        assert restored.inherit_variables is True
        assert restored.exit_code_variable == "rc"
        assert restored.cost_variable == "cost"

    def test_roundtrip_metric_variables(self):
        original = SpawnBlock(
            id="sp2",
            name="Spawn",
            agent_name="cell-1",
            command_name="tb-cell",
            duration_variable="dur_1",
            input_tokens_variable="in_1",
            output_tokens_variable="out_1",
            cache_creation_tokens_variable="cc_1",
            cache_read_tokens_variable="cr_1",
        )
        restored = BlockAdapter.validate_python(original.model_dump())
        assert isinstance(restored, SpawnBlock)
        assert restored.duration_variable == "dur_1"
        assert restored.input_tokens_variable == "in_1"
        assert restored.output_tokens_variable == "out_1"
        assert restored.cache_creation_tokens_variable == "cc_1"
        assert restored.cache_read_tokens_variable == "cr_1"

    def test_roundtrip_cwd(self):
        original = SpawnBlock(
            id="sp3",
            name="Spawn",
            agent_name="cell-1",
            command_name="tb-cell",
            cwd="{{base}}/cell-1",
        )
        restored = BlockAdapter.validate_python(original.model_dump())
        assert isinstance(restored, SpawnBlock)
        assert restored.cwd == "{{base}}/cell-1"


class TestWaitBlock:
    def test_creation(self):
        b = WaitBlock(
            name="Wait All",
            wait_for=["worker-1", "worker-2"],
            timeout_seconds=300,
        )
        assert b.type == BlockType.WAIT
        assert b.wait_for == ["worker-1", "worker-2"]
        assert b.timeout_seconds == 300

    def test_defaults(self):
        b = WaitBlock()
        assert b.wait_for == []
        assert b.timeout_seconds is None

    def test_deserialize(self):
        b = BlockAdapter.validate_python({
            "type": "wait",
            "name": "W",
            "wait_for": ["a1"],
        })
        assert isinstance(b, WaitBlock)
        assert b.wait_for == ["a1"]

    def test_roundtrip_json(self):
        original = WaitBlock(
            id="w1",
            name="Wait",
            wait_for=["a", "b"],
            timeout_seconds=60,
        )
        json_str = original.model_dump_json()
        restored = BlockAdapter.validate_json(json_str)
        assert isinstance(restored, WaitBlock)
        assert restored.wait_for == ["a", "b"]
        assert restored.timeout_seconds == 60


class TestExitBlock:
    def test_creation(self):
        b = ExitBlock(name="Bail", exit_code=1, exit_message="Failed")
        assert b.type == BlockType.EXIT
        assert b.exit_code == 1
        assert b.exit_message == "Failed"

    def test_defaults(self):
        b = ExitBlock()
        assert b.exit_code == 0
        assert b.exit_message == ""

    def test_deserialize(self):
        b = BlockAdapter.validate_python({
            "type": "exit",
            "name": "X",
            "exit_code": 2,
        })
        assert isinstance(b, ExitBlock)
        assert b.exit_code == 2

    def test_roundtrip(self):
        original = ExitBlock(id="x1", name="Exit", exit_code=42, exit_message="done")
        data = original.model_dump()
        restored = BlockAdapter.validate_python(data)
        assert isinstance(restored, ExitBlock)
        assert restored.exit_code == 42
        assert restored.exit_message == "done"


class TestInputBlock:
    def test_creation(self):
        b = InputBlock(name="Get Input", output_variable="user_response")
        assert b.type == BlockType.INPUT
        assert b.output_variable == "user_response"

    def test_defaults(self):
        b = InputBlock()
        assert b.output_variable is None

    def test_deserialize(self):
        b = BlockAdapter.validate_python({
            "type": "input",
            "name": "I",
            "output_variable": "answer",
        })
        assert isinstance(b, InputBlock)
        assert b.output_variable == "answer"

    def test_roundtrip_json(self):
        original = InputBlock(id="i1", name="Ask", output_variable="resp")
        json_str = original.model_dump_json()
        restored = BlockAdapter.validate_json(json_str)
        assert isinstance(restored, InputBlock)
        assert restored.output_variable == "resp"
