"""Tests for CLI argument parsing."""

import pytest
from flowcoder_engine.cli import build_inner_claude_cmd, build_variables, parse_args
from flowcoder_flowchart import Argument


class TestInnerClaudeCmd:
    """The inner CLI must be launchable, not merely well-formed.

    Claude CLI refuses `-p` with `--output-format stream-json` unless
    --verbose is present, exiting immediately.  That produces EOF on
    stdout, so read() returns None and queries come back empty — a
    silent wrong answer, which is why these assertions are explicit
    rather than left to an integration test to notice.
    """

    def test_verbose_present_without_the_flag(self):
        """The bug: a host that never passes --verbose still needs it."""
        cmd = build_inner_claude_cmd(parse_args([]), "/usr/bin/claude")
        assert "--verbose" in cmd, (
            "inner CLI would exit immediately: -p with "
            "--output-format stream-json requires --verbose"
        )

    def test_verbose_present_with_the_flag(self):
        cmd = build_inner_claude_cmd(parse_args(["--verbose"]), "/usr/bin/claude")
        assert "--verbose" in cmd

    def test_verbose_not_duplicated_when_host_passes_it(self):
        """Our parser absorbs --verbose so it cannot arrive twice."""
        cmd = build_inner_claude_cmd(parse_args(["--verbose"]), "/usr/bin/claude")
        assert cmd.count("--verbose") == 1

    def test_protocol_trio_always_present(self):
        """--verbose is only required because of these two; pin all three."""
        cmd = build_inner_claude_cmd(parse_args([]), "/usr/bin/claude")
        assert "-p" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"
        assert cmd[cmd.index("--input-format") + 1] == "stream-json"

    def test_survives_passthrough_from_a_host(self):
        """A host forwarding raw flags must still get a launchable command."""
        cmd = build_inner_claude_cmd(
            parse_args(["--", "--include-partial-messages"]), "/usr/bin/claude"
        )
        assert "--verbose" in cmd
        assert "--include-partial-messages" in cmd


class TestParseArgs:
    def test_no_args(self):
        """Engine starts with no arguments (proxy mode)."""
        args = parse_args([])
        assert args.claude_path is None
        assert args.search_paths is None
        assert args.max_blocks == 1000
        assert args.passthrough == []

    def test_with_claude_path(self):
        args = parse_args(["--claude-path", "/usr/local/bin/claude"])
        assert args.claude_path == "/usr/local/bin/claude"

    def test_with_search_paths(self):
        args = parse_args(["--search-path", "/path1", "--search-path", "/path2"])
        assert args.search_paths == ["/path1", "/path2"]

    def test_max_blocks(self):
        args = parse_args(["--max-blocks", "500"])
        assert args.max_blocks == 500

    def test_passthrough_args(self):
        """Unknown args are collected for pass-through to inner claude."""
        args = parse_args(["--some-unknown-flag", "value"])
        assert "--some-unknown-flag" in args.passthrough
        assert "value" in args.passthrough

    def test_known_args_not_in_passthrough(self):
        """Explicitly parsed args are NOT in passthrough."""
        args = parse_args(["--model", "haiku", "--verbose"])
        assert args.model == "haiku"
        assert args.verbose is True
        assert "--model" not in args.passthrough

    def test_mixed_own_and_passthrough(self):
        args = parse_args([
            "--search-path", "/cmds",
            "--model", "opus",
            "--max-blocks", "50",
            "--system-prompt", "test",
        ])
        assert args.search_paths == ["/cmds"]
        assert args.max_blocks == 50
        assert args.model == "opus"
        assert args.system_prompt == "test"

class TestBuildVariables:
    def test_from_args_string(self):
        declared = [Argument(name="file"), Argument(name="mode", required=False, default="strict")]
        result = build_variables("main.py", declared)
        assert result["$1"] == "main.py"
        assert result["file"] == "main.py"
        assert result["$2"] == "strict"
        assert result["mode"] == "strict"

    def test_missing_required(self):
        declared = [Argument(name="file")]
        with pytest.raises(ValueError, match="Missing required"):
            build_variables("", declared)

    def test_empty_no_args(self):
        result = build_variables("", [])
        assert result == {}

    def test_quoted_args(self):
        declared = [Argument(name="msg")]
        result = build_variables('"hello world"', declared)
        assert result["$1"] == "hello world"
        assert result["msg"] == "hello world"

    def test_extra_positional(self):
        declared = [Argument(name="first")]
        result = build_variables("a b c", declared)
        assert result["$1"] == "a"
        assert result["first"] == "a"
        assert result["$2"] == "b"
        assert result["$3"] == "c"


class TestAnswerJsonSchema:
    """`--json-schema` names the shape of the flowchart's FINAL answer.

    It must never reach the inner CLI. The inner command is built once and
    reused for every block (session.py), so a CLI-level --json-schema would
    constrain every turn -- including control blocks that carry their own
    `output_schema`. Observed in production: a host passed a task schema of
    {digits, count}; the CLASSIFY block's {isTask, hasRefTopics, refFiles}
    was replaced by it, `isTask` was never set, the branch went falsy, and
    the entire task path was skipped. The flowchart silently stopped doing
    the work it was asked to do.
    """

    SCHEMA = '{"type":"object","properties":{"digits":{"type":"string"}}}'

    def test_schema_is_parsed_as_a_first_class_flag(self):
        args = parse_args(["--json-schema", self.SCHEMA])
        assert args.json_schema == self.SCHEMA

    def test_schema_does_not_leak_into_passthrough(self):
        """Passthrough is appended verbatim to the inner command."""
        args = parse_args(["--json-schema", self.SCHEMA])
        assert "--json-schema" not in args.passthrough

    def test_schema_never_reaches_the_inner_cli(self):
        cmd = build_inner_claude_cmd(
            parse_args(["--json-schema", self.SCHEMA]), "/usr/bin/claude"
        )
        assert "--json-schema" not in cmd, (
            "a session-wide schema overrides every block's own output_schema"
        )
        assert self.SCHEMA not in cmd

    def test_absent_by_default(self):
        assert parse_args([]).json_schema is None

    def test_other_passthrough_flags_still_pass_through(self):
        """Narrow fix: only --json-schema is intercepted."""
        args = parse_args(["--json-schema", self.SCHEMA, "--some-future-flag", "x"])
        cmd = build_inner_claude_cmd(args, "/usr/bin/claude")
        assert "--some-future-flag" in cmd and "x" in cmd
