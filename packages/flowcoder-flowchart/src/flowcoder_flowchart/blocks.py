"""Block types for flowchart workflows.

Defines the BlockType enum, all block models, and the Block discriminated union.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BlockType(StrEnum):
    START = "start"
    END = "end"
    PROMPT = "prompt"
    BRANCH = "branch"
    VARIABLE = "variable"
    BASH = "bash"
    COMMAND = "command"
    REFRESH = "refresh"
    SPAWN = "spawn"
    WAIT = "wait"
    EXIT = "exit"
    INPUT = "input"


# Pride compat: map int/float to number
_VARIABLE_TYPE_ALIASES: dict[str, str] = {"int": "number", "float": "number"}


class VariableType(StrEnum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"


class Position(BaseModel):
    x: float
    y: float


class BlockBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    session: str = "default"
    position: Position | None = None
    # Any block the walker can time out needs this, not just prompts: the
    # input block's ReadTimeoutError handler reports it too.
    timeout_seconds: int = 21600


class StartBlock(BlockBase):
    type: Literal[BlockType.START] = BlockType.START


class EndBlock(BlockBase):
    type: Literal[BlockType.END] = BlockType.END


class PromptBlock(BlockBase):
    type: Literal[BlockType.PROMPT] = BlockType.PROMPT
    prompt: str
    output_variable: str | None = None
    output_schema: dict[str, Any] | None = None


class BranchBlock(BlockBase):
    type: Literal[BlockType.BRANCH] = BlockType.BRANCH
    condition: str


class VariableBlock(BlockBase):
    type: Literal[BlockType.VARIABLE] = BlockType.VARIABLE
    variable_name: str
    variable_value: str = ""
    variable_type: VariableType = VariableType.STRING

    @field_validator("variable_type", mode="before")
    @classmethod
    def normalize_variable_type(cls, v: str) -> str:
        return _VARIABLE_TYPE_ALIASES.get(v, v)


class BashBlock(BlockBase):
    type: Literal[BlockType.BASH] = BlockType.BASH
    command: str
    capture_output: bool = True
    output_variable: str | None = None
    output_type: VariableType = VariableType.STRING
    continue_on_error: bool = False
    working_directory: str | None = None
    exit_code_variable: str | None = None

    @field_validator("output_type", mode="before")
    @classmethod
    def normalize_output_type(cls, v: str) -> str:
        return _VARIABLE_TYPE_ALIASES.get(v, v)


class CommandBlock(BlockBase):
    type: Literal[BlockType.COMMAND] = BlockType.COMMAND
    command_name: str
    arguments: str = ""
    inherit_variables: bool = False
    merge_output: bool = False


class RefreshBlock(BlockBase):
    type: Literal[BlockType.REFRESH] = BlockType.REFRESH
    target_session: str | None = None


class SpawnBlock(BlockBase):
    """Spawn a named agent sub-session running a command asynchronously."""

    type: Literal[BlockType.SPAWN] = BlockType.SPAWN
    agent_name: str = ""
    command_name: str = ""
    arguments: str = ""
    inherit_variables: bool = False
    exit_code_variable: str | None = None
    cost_variable: str | None = None
    duration_variable: str | None = None
    input_tokens_variable: str | None = None
    output_tokens_variable: str | None = None
    cache_creation_tokens_variable: str | None = None
    cache_read_tokens_variable: str | None = None
    config_file: str | None = None
    model: str | None = None
    backend: str | None = None
    cwd: str | None = None


class WaitBlock(BlockBase):
    """Wait for one or more spawned agent sessions to complete."""

    type: Literal[BlockType.WAIT] = BlockType.WAIT
    wait_for: list[str] = Field(default_factory=list)
    timeout_seconds: int | None = None


class ExitBlock(BlockBase):
    """Explicitly exit the flowchart with a given exit code."""

    type: Literal[BlockType.EXIT] = BlockType.EXIT
    exit_code: int = 0
    exit_message: str = ""


class InputBlock(BlockBase):
    """Pause the flowchart, accept user input, send to the agent session."""

    type: Literal[BlockType.INPUT] = BlockType.INPUT
    output_variable: str | None = None


Block = Annotated[
    StartBlock
    | EndBlock
    | PromptBlock
    | BranchBlock
    | VariableBlock
    | BashBlock
    | CommandBlock
    | RefreshBlock
    | SpawnBlock
    | WaitBlock
    | ExitBlock
    | InputBlock,
    Field(discriminator="type"),
]
