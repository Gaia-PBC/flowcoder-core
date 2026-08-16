"""End-to-end tests for the ``flowcoder`` terminal runner.

Runs the real console-script entry point as a subprocess against the stub
``claude`` from ``_stub_claude.py``, so the whole path — argument parsing,
command resolution, session start-up, the walker, and terminal rendering — is
exercised at zero token cost.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from flowcoder_flowchart import (
    BashBlock,
    Command,
    Connection,
    EndBlock,
    ExitBlock,
    Flowchart,
    InputBlock,
    PromptBlock,
    StartBlock,
    save_command,
)

from .engine_harness import STUB_CLAUDE

# The stub answers instantly, so a healthy run finishes in well under a second;
# this only bounds a *broken* one (e.g. an input block waiting on stdin forever).
RUN_TIMEOUT = 30


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A command directory plus a launcher that stands in for the claude CLI."""
    commands = tmp_path / "commands"
    commands.mkdir()

    launcher = tmp_path / "_claude_launcher.sh"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{STUB_CLAUDE}" "$@"\n')
    launcher.chmod(0o755)

    _save(
        commands,
        "demo",
        Flowchart(
            blocks={
                "s": StartBlock(id="s", name="START"),
                "b": BashBlock(
                    id="b", name="ECHO", command="echo shell-ran",
                    capture_output=True, output_variable="shell",
                ),
                "p": PromptBlock(id="p", name="ASK", prompt="greet $1"),
                "e": EndBlock(id="e", name="END"),
            },
            connections=[
                Connection(source_id="s", target_id="b"),
                Connection(source_id="b", target_id="p"),
                Connection(source_id="p", target_id="e"),
            ],
        ),
    )
    _save(
        commands,
        "asks",
        Flowchart(
            blocks={
                "s": StartBlock(id="s", name="START"),
                "i": InputBlock(id="i", name="ASK-USER", timeout_seconds=20),
                "e": EndBlock(id="e", name="END"),
            },
            connections=[
                Connection(source_id="s", target_id="i"),
                Connection(source_id="i", target_id="e"),
            ],
        ),
    )
    _save(
        commands,
        "bails",
        Flowchart(
            blocks={
                "s": StartBlock(id="s", name="START"),
                "x": ExitBlock(id="x", name="BAIL", exit_code=3, exit_message="bailing"),
            },
            connections=[Connection(source_id="s", target_id="x")],
        ),
    )
    return tmp_path


def _save(directory: Path, name: str, flowchart: Flowchart) -> None:
    save_command(
        Command(id=name, name=name, flowchart=flowchart), directory / f"{name}.json"
    )


def run_cli(
    workspace: Path,
    *args: str,
    stdin: str = "",
    stdin_file: Path | None = None,
    devnull: bool = False,
) -> subprocess.CompletedProcess:
    """Invoke the runner exactly as the ``flowcoder`` console script does.

    The three stdin forms are not interchangeable: a pipe is pollable and gets
    an event-loop reader, while ``/dev/null`` and a redirected file are not and
    take the direct-read path instead.
    """
    command = [
        sys.executable, "-m", "flowcoder_engine.runner",
        "--claude-path", str(workspace / "_claude_launcher.sh"),
        "--search-path", str(workspace / "commands"),
        "--no-color",
        *args,
    ]
    kwargs: dict = {"input": stdin}
    handle = None
    if devnull:
        kwargs = {"stdin": subprocess.DEVNULL}
    elif stdin_file is not None:
        handle = stdin_file.open()
        kwargs = {"stdin": handle}
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
            cwd=str(workspace),
            **kwargs,
        )
    finally:
        if handle is not None:
            handle.close()


def test_runs_a_flowchart_and_renders_its_progress(workspace):
    result = run_cli(workspace, "demo", "World")
    assert result.returncode == 0, result.stderr
    # Block names and the assistant's reply, not raw protocol JSON.
    assert "ASK" in result.stdout
    assert "stub reply to: greet World" in result.stdout
    assert "completed" in result.stdout
    assert '"type":' not in result.stdout


def test_a_leading_slash_on_the_command_is_accepted(workspace):
    assert run_cli(workspace, "/demo", "World").returncode == 0


def test_json_mode_puts_the_variables_on_stdout_alone(workspace):
    result = run_cli(workspace, "--json", "demo", "World")
    assert result.returncode == 0, result.stderr
    variables = json.loads(result.stdout)
    assert variables["$1"] == "World"
    assert variables["shell"] == "shell-ran"
    # Progress moved to stderr so the document is machine-readable.
    assert "ASK" in result.stderr


def test_quoted_arguments_survive_as_one_argument(workspace):
    result = run_cli(workspace, "--json", "demo", "two words")
    assert json.loads(result.stdout)["$1"] == "two words"


def test_input_blocks_read_a_line_from_stdin(workspace):
    result = run_cli(workspace, "asks", stdin="typed answer\n")
    assert result.returncode == 0, result.stderr
    assert "stub reply to: typed answer" in result.stdout


def test_an_exit_block_sets_the_process_exit_code(workspace):
    result = run_cli(workspace, "bails")
    assert result.returncode == 3
    assert "exited (3)" in result.stdout


def test_an_unknown_command_fails_with_the_paths_it_searched(workspace):
    result = run_cli(workspace, "nope")
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()
    assert "commands" in result.stderr


def test_list_shows_the_resolvable_commands(workspace):
    result = run_cli(workspace, "--list")
    assert result.returncode == 0
    assert "demo" in result.stdout and "asks" in result.stdout


def test_devnull_stdin_does_not_crash_the_event_loop(workspace):
    """/dev/null is not pollable; attaching a reader to it must not be tried.

    The failure it guards against is asynchronous — the selector rejects the fd
    inside a loop callback — so it surfaces as a traceback on stderr beside an
    otherwise successful run, not as a non-zero exit.
    """
    result = run_cli(workspace, "demo", "World", devnull=True)
    assert result.returncode == 0
    assert "Traceback" not in result.stderr
    assert "PermissionError" not in result.stderr


def test_input_blocks_also_read_from_a_redirected_file(workspace):
    """A regular file is not pollable either, so it takes the direct-read path."""
    answers = workspace / "answers.txt"
    answers.write_text("answer from a file\n")
    result = run_cli(workspace, "asks", stdin_file=answers)
    assert result.returncode == 0, result.stderr
    assert "stub reply to: answer from a file" in result.stdout
    assert "Traceback" not in result.stderr
