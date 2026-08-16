"""Unit tests for the ``flowcoder`` terminal runner."""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from flowcoder_engine.runner import (
    DEFAULT_PERMISSION_MODE,
    PermissionPrompter,
    Style,
    TerminalProtocol,
    build_runner_claude_cmd,
    list_commands,
    parse_runner_args,
)
from flowcoder_flowchart import Command, Connection, EndBlock, Flowchart, StartBlock, save_command

PLAIN = Style(enabled=False)


def make_protocol(verbose: bool = False) -> tuple[TerminalProtocol, io.StringIO]:
    stream = io.StringIO()
    return TerminalProtocol(style=PLAIN, stream=stream, verbose=verbose), stream


# ── argument parsing ──────────────────────────────────────────────────


def test_flags_before_command_are_the_runners():
    args = parse_runner_args(["--model", "sonnet", "--search-path", "./cmds", "deploy"])
    assert args.model == "sonnet"
    assert args.search_paths == ["./cmds"]
    assert args.command == "deploy"
    assert args.args == []


def test_everything_after_the_command_belongs_to_the_flowchart():
    """REMAINDER, so a flowchart argument that looks like a flag stays one."""
    args = parse_runner_args(["deploy", "--verbose", "prod", "-x"])
    assert args.command == "deploy"
    assert args.args == ["--verbose", "prod", "-x"]
    assert args.verbose is False  # consumed by the flowchart, not the runner


def test_permission_mode_defaults_to_bypass_but_is_overridable():
    assert parse_runner_args(["deploy"]).permission_mode == DEFAULT_PERMISSION_MODE
    assert parse_runner_args(["--permission-mode", "plan", "deploy"]).permission_mode == "plan"


def test_passthrough_is_defined_for_build_inner_claude_cmd():
    """build_inner_claude_cmd reads args.passthrough; it must always exist."""
    assert parse_runner_args(["deploy"]).passthrough == []


def test_missing_command_without_list_is_an_error():
    with pytest.raises(SystemExit):
        parse_runner_args([])


def test_list_needs_no_command():
    assert parse_runner_args(["--list"]).list_commands is True


# ── inner claude command ──────────────────────────────────────────────


def test_bypass_mode_does_not_ask_so_no_prompt_tool_is_wired():
    cmd = build_runner_claude_cmd(parse_runner_args(["deploy"]), "/bin/claude")
    assert "--permission-prompt-tool" not in cmd
    assert "--permission-mode" in cmd and DEFAULT_PERMISSION_MODE in cmd


def test_a_prompting_mode_gets_the_permission_prompt_tool():
    """Without this flag Claude denies un-approved tools without ever asking."""
    cmd = build_runner_claude_cmd(
        parse_runner_args(["--permission-mode", "default", "deploy"]), "/bin/claude"
    )
    assert cmd[-2:] == ["--permission-prompt-tool", "stdio"]


def test_claude_settings_still_reach_the_inner_command():
    cmd = build_runner_claude_cmd(
        parse_runner_args(["--model", "haiku", "--cwd", "/tmp", "deploy"]), "/bin/claude"
    )
    assert cmd[0] == "/bin/claude"
    assert cmd[cmd.index("--model") + 1] == "haiku"


# ── rendering ─────────────────────────────────────────────────────────


def test_block_start_is_rendered_not_emitted_as_json():
    protocol, stream = make_protocol()
    protocol.emit_block_start("b1", "DEPLOY", "prompt")
    out = stream.getvalue()
    assert "DEPLOY" in out and "prompt" in out
    assert "block_start" not in out  # rendered, not dumped


def test_assistant_text_is_printed_and_tool_calls_are_summarized():
    protocol, stream = make_protocol()
    protocol.emit_forwarded(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Here is the answer"},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}},
                ]
            },
        },
        "main",
        "b1",
        "ASK",
    )
    out = stream.getvalue()
    assert "Here is the answer" in out
    assert "Bash(ls -la)" in out


def test_successful_blocks_stay_quiet_but_failures_do_not():
    protocol, stream = make_protocol()
    protocol.emit_block_complete("b1", "OK-BLOCK", True)
    assert "OK-BLOCK" not in stream.getvalue()
    protocol.emit_block_complete("b2", "BAD-BLOCK", False)
    assert "BAD-BLOCK" in stream.getvalue()


def test_verbose_shows_block_completions_and_engine_logs():
    protocol, stream = make_protocol(verbose=True)
    protocol.emit_block_complete("b1", "OK-BLOCK", True)
    assert "OK-BLOCK" in stream.getvalue()


def test_quiet_engine_logs_are_suppressed(capsys):
    protocol, _ = make_protocol()
    protocol.log("inner claude started")
    assert "inner claude started" not in capsys.readouterr().err


def test_flowchart_complete_is_left_to_the_runners_summary():
    """The runner prints its own summary from ExecutionResult; don't double up."""
    protocol, stream = make_protocol()
    protocol.emit_flowchart_complete(status="completed", duration_ms=10, cost_usd=1.0)
    assert stream.getvalue() == ""


async def test_input_request_answers_the_walker_from_stdin():
    """An input block must get its input_response pushed into the inbox."""

    class Scripted(TerminalProtocol):
        async def read_line(self, prompt: str) -> str:
            return "typed answer"

    protocol = Scripted(style=PLAIN, stream=io.StringIO(), verbose=False)
    protocol.emit_system("input_request", {"block_id": "b1", "block_name": "ASK"})
    message = await protocol.read_message()
    assert message == {
        "type": "input_response",
        "block_id": "b1",
        "content": "typed answer",
    }
    await protocol.stop()


async def test_stop_cancels_a_prompt_still_waiting():
    """A run that ends mid-prompt must not leave the task pending."""

    class Hanging(TerminalProtocol):
        async def read_line(self, prompt: str) -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    protocol = Hanging(style=PLAIN, stream=io.StringIO(), verbose=False)
    protocol.emit_system("input_request", {"block_id": "b1", "block_name": "ASK"})
    await asyncio.sleep(0)  # let the task start
    await protocol.stop()
    assert protocol._pending == set()


# ── tool permissions ──────────────────────────────────────────────────


class _Answering(TerminalProtocol):
    """A protocol whose prompts return canned answers."""

    def __init__(self, answers: list[str]) -> None:
        super().__init__(style=PLAIN, stream=io.StringIO(), verbose=False)
        self.answers = answers

    async def read_line(self, prompt: str) -> str:
        return self.answers.pop(0)


def _request(tool: str = "Bash", request_id: str = "r1") -> dict:
    return {
        "type": "control_request",
        "request_id": request_id,
        "request": {"subtype": "can_use_tool", "tool_name": tool, "input": {"command": "ls"}},
    }


async def test_yes_allows_and_keeps_the_input():
    protocol = _Answering(["y"])
    response = await PermissionPrompter(protocol, PLAIN, assume_yes=False)(_request())
    assert response["response"]["request_id"] == "r1"
    assert response["response"]["response"]["behavior"] == "allow"
    assert response["response"]["response"]["updatedInput"] == {"command": "ls"}


async def test_no_denies():
    protocol = _Answering(["n"])
    response = await PermissionPrompter(protocol, PLAIN, assume_yes=False)(_request())
    assert response["response"]["response"]["behavior"] == "deny"


async def test_empty_answer_denies():
    """Enter-on-the-prompt must not be read as consent."""
    protocol = _Answering([""])
    response = await PermissionPrompter(protocol, PLAIN, assume_yes=False)(_request())
    assert response["response"]["response"]["behavior"] == "deny"


async def test_always_stops_asking_for_that_tool():
    protocol = _Answering(["a"])  # one answer only: a second prompt would IndexError
    prompter = PermissionPrompter(protocol, PLAIN, assume_yes=False)
    assert (await prompter(_request()))["response"]["response"]["behavior"] == "allow"
    assert (await prompter(_request()))["response"]["response"]["behavior"] == "allow"
    assert protocol.answers == []


async def test_always_is_scoped_to_the_tool_it_was_given_for():
    protocol = _Answering(["a", "n"])
    prompter = PermissionPrompter(protocol, PLAIN, assume_yes=False)
    await prompter(_request("Bash"))
    response = await prompter(_request("Write"))
    assert response["response"]["response"]["behavior"] == "deny"


async def test_assume_yes_never_prompts():
    protocol = _Answering([])  # no answers available at all
    response = await PermissionPrompter(protocol, PLAIN, assume_yes=True)(_request())
    assert response["response"]["response"]["behavior"] == "allow"


async def test_other_control_requests_are_acknowledged_not_left_hanging():
    protocol = _Answering([])
    request = {"request_id": "r9", "request": {"subtype": "hook_callback"}}
    response = await PermissionPrompter(protocol, PLAIN, assume_yes=False)(request)
    assert response["type"] == "control_response"
    assert response["response"]["subtype"] == "success"
    assert response["response"]["request_id"] == "r9"


# ── --list ────────────────────────────────────────────────────────────


def _write_command(directory, name: str, description: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    flowchart = Flowchart(
        blocks={"s": StartBlock(id="s", name="START"), "e": EndBlock(id="e", name="END")},
        connections=[Connection(source_id="s", target_id="e")],
    )
    save_command(
        Command(id=name, name=name, description=description, flowchart=flowchart),
        directory / f"{name}.json",
    )


@pytest.fixture
def isolated_search(tmp_path, monkeypatch):
    """Cut the two implicit search roots — cwd and $HOME — out of the test.

    ``_command_dirs`` always includes ``./commands`` and ``~/.flowcoder/commands``,
    so without this a developer's own commands leak into the assertions.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_list_reports_commands_with_descriptions(isolated_search):
    _write_command(isolated_search / "cmds", "deploy", "Ship it")
    stream = io.StringIO()
    assert list_commands([str(isolated_search / "cmds")], PLAIN, stream) == 0
    out = stream.getvalue()
    assert "deploy" in out and "Ship it" in out


def test_list_ignores_json_that_is_not_a_command(isolated_search):
    directory = isolated_search / "cmds"
    _write_command(directory, "deploy")
    (directory / "settings.json").write_text(json.dumps({"unrelated": True}))
    stream = io.StringIO()
    list_commands([str(directory)], PLAIN, stream)
    assert "settings" not in stream.getvalue()


def test_list_without_any_commands_is_a_nonzero_exit(isolated_search):
    stream = io.StringIO()
    assert list_commands([str(isolated_search / "empty")], PLAIN, stream) == 1
    assert "No commands found" in stream.getvalue()
