"""Does a locally served model actually drive a flowchart?

The stub tiers prove the engine's plumbing; they say nothing about whether the
model on the other end can hold up its end of the contract.  These tests answer
the questions a testbench actually cares about, in order of how likely they are
to fail on a small local model:

1. does a prompt block come back at all,
2. does ``output_schema`` produce parseable JSON that lands in variables,
3. does a branch driven by that variable take the right path.

They go through the ``flowcoder`` CLI as a subprocess, so what is measured is
the same path the Docker image runs, not an in-process approximation.

Deselected unless ANTHROPIC_BASE_URL is set — see conftest.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from flowcoder_flowchart import (
    BranchBlock,
    Command,
    Connection,
    EndBlock,
    Flowchart,
    PromptBlock,
    StartBlock,
    VariableBlock,
    save_command,
)

from .conftest import run_timeout


@pytest.fixture
def commands(tmp_path: Path) -> Path:
    directory = tmp_path / "commands"
    directory.mkdir()
    return directory


def save(directory: Path, name: str, flowchart: Flowchart) -> None:
    save_command(
        Command(id=name, name=name, flowchart=flowchart), directory / f"{name}.json"
    )


def run(commands: Path, model: str | None, name: str, *args: str) -> dict:
    """Run one command against the local endpoint; return its final variables.

    The endpoint itself comes from ANTHROPIC_BASE_URL in the inherited
    environment — the same wiring a Docker run uses.
    """
    command = [
        sys.executable, "-m", "flowcoder_engine.runner",
        "--search-path", str(commands),
        "--cwd", str(commands.parent),
        "--no-color",
        "--json",
    ]
    if model:
        command += ["--model", model]
    result = subprocess.run(
        [*command, name, *args],
        capture_output=True,
        text=True,
        timeout=run_timeout(),
    )
    assert result.returncode == 0, (
        f"flowcoder exited {result.returncode}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
    return json.loads(result.stdout)


def test_a_prompt_block_gets_an_answer(commands, model_name):
    """The floor: the model is reachable and the session completes a turn."""
    save(
        commands,
        "answer",
        Flowchart(
            blocks={
                "s": StartBlock(id="s", name="START"),
                "p": PromptBlock(
                    id="p", name="ASK",
                    prompt="Reply with the single word: pong. Do not use any tools.",
                    output_variable="reply",
                ),
                "e": EndBlock(id="e", name="END"),
            },
            connections=[
                Connection(source_id="s", target_id="p"),
                Connection(source_id="p", target_id="e"),
            ],
        ),
    )
    variables = run(commands, model_name, "answer")
    assert "pong" in str(variables.get("reply", "")).lower()


def test_structured_output_lands_in_variables(commands, model_name):
    """output_schema is where small models fail first.

    The walker appends the schema to the prompt and feeds the reply through
    parse_json_from_response, so a model that wraps its JSON in commentary is
    tolerated — one that does not emit JSON at all is not, and that is exactly
    what a bench needs to surface.
    """
    save(
        commands,
        "structured",
        Flowchart(
            blocks={
                "s": StartBlock(id="s", name="START"),
                "p": PromptBlock(
                    id="p", name="COMPUTE",
                    prompt="What is 2 + 2? Do not use any tools.",
                    output_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "number"}},
                        "required": ["answer"],
                    },
                ),
                "e": EndBlock(id="e", name="END"),
            },
            connections=[
                Connection(source_id="s", target_id="p"),
                Connection(source_id="p", target_id="e"),
            ],
        ),
    )
    variables = run(commands, model_name, "structured")
    assert "answer" in variables, f"no parsed JSON in {variables}"
    assert float(variables["answer"]) == 4


def test_a_branch_follows_the_models_answer(commands, model_name):
    """The full loop: model → parsed variable → control flow.

    A flowchart is only worth running if its branches are steerable, so this is
    the test that decides whether a given local model is usable at all.
    """
    save(
        commands,
        "branching",
        Flowchart(
            blocks={
                "s": StartBlock(id="s", name="START"),
                "p": PromptBlock(
                    id="p", name="JUDGE",
                    prompt="Is 10 greater than 3? Do not use any tools.",
                    output_schema={
                        "type": "object",
                        "properties": {"greater": {"type": "boolean"}},
                        "required": ["greater"],
                    },
                ),
                "b": BranchBlock(id="b", name="GREATER?", condition="greater"),
                "yes": VariableBlock(
                    id="yes", name="YES", variable_name="outcome", variable_value="yes"
                ),
                "no": VariableBlock(
                    id="no", name="NO", variable_name="outcome", variable_value="no"
                ),
                "e": EndBlock(id="e", name="END"),
            },
            connections=[
                Connection(source_id="s", target_id="p"),
                Connection(source_id="p", target_id="b"),
                Connection(source_id="b", target_id="yes", is_true_path=True),
                Connection(source_id="b", target_id="no", is_true_path=False),
                Connection(source_id="yes", target_id="e"),
                Connection(source_id="no", target_id="e"),
            ],
        ),
    )
    variables = run(commands, model_name, "branching")
    assert variables.get("outcome") == "yes", (
        f"branch went the wrong way; model said greater={variables.get('greater')!r}"
    )


def test_command_arguments_reach_the_prompt(commands, model_name):
    """$1 substitution, end to end, with a real model reading the result."""
    save(
        commands,
        "echoes",
        Flowchart(
            blocks={
                "s": StartBlock(id="s", name="START"),
                "p": PromptBlock(
                    id="p", name="REPEAT",
                    prompt="Reply with exactly this word and nothing else: $1. "
                           "Do not use any tools.",
                    output_variable="reply",
                ),
                "e": EndBlock(id="e", name="END"),
            },
            connections=[
                Connection(source_id="s", target_id="p"),
                Connection(source_id="p", target_id="e"),
            ],
        ),
    )
    variables = run(commands, model_name, "echoes", "kumquat")
    assert variables["$1"] == "kumquat"
    assert "kumquat" in str(variables.get("reply", "")).lower()
