"""Tests for Pride flowcoder format compatibility.

Verifies that:
- Pride-format JSON (source_block_id, sound_effect, int types) parses correctly
- Branch expression evaluation works (==, !=, <, >, >=, <=, !var)
- Variable type aliases (int/float -> number) work end-to-end
- All Pride example flowcharts load without error
"""

from pathlib import Path

import pytest
from flowcoder_engine.walker import GraphWalker, _evaluate_condition
from flowcoder_flowchart import (
    BranchBlock,
    Connection,
    EndBlock,
    Flowchart,
    StartBlock,
    load_command,
)

from tests.conftest import MockProtocol, MockSession

# Pride example flowcharts are vendored into the repo's own commands/ dir.
# (Was ~/pride-flowcoder/commands — an external-install path that no longer
# exists, which silently skipped every example-load test. Repo layout:
# <repo>/packages/flowcoder-engine/tests/unit/this_file -> parents[4] == <repo>.)
PRIDE_COMMANDS_DIR = Path(__file__).resolve().parents[4] / "commands"


# ------------------------------------------------------------------
# Branch expression evaluation
# ------------------------------------------------------------------

class TestBranchExpressions:
    """Test the _evaluate_condition() helper directly."""

    def test_simple_truthy(self):
        assert _evaluate_condition("flag", {"flag": True}) is True

    def test_simple_falsy(self):
        assert _evaluate_condition("flag", {"flag": False}) is False

    def test_missing_var_is_falsy(self):
        assert _evaluate_condition("nonexistent", {}) is False

    def test_negation_true(self):
        assert _evaluate_condition("!hasErrors", {"hasErrors": False}) is True

    def test_negation_false(self):
        assert _evaluate_condition("!hasErrors", {"hasErrors": True}) is False

    def test_equality_numeric(self):
        assert _evaluate_condition("exitCode == 0", {"exitCode": 0}) is True

    def test_inequality_numeric(self):
        assert _evaluate_condition("exitCode != 0", {"exitCode": 1}) is True

    def test_less_than(self):
        assert _evaluate_condition("i < 5", {"i": 2}) is True
        assert _evaluate_condition("i < 5", {"i": 5}) is False
        assert _evaluate_condition("i < 5", {"i": 8}) is False

    def test_greater_than(self):
        assert _evaluate_condition("i > 5", {"i": 8}) is True
        assert _evaluate_condition("i > 5", {"i": 5}) is False

    def test_less_equal(self):
        assert _evaluate_condition("i <= 5", {"i": 5}) is True

    def test_greater_equal(self):
        assert _evaluate_condition("i >= 5", {"i": 5}) is True

    def test_string_equality(self):
        assert _evaluate_condition('status == "done"', {"status": "done"}) is True
        assert _evaluate_condition('status == "done"', {"status": "pending"}) is False

    def test_string_inequality(self):
        assert _evaluate_condition('status != "done"', {"status": "pending"}) is True

    def test_zero_string_is_falsy(self):
        assert _evaluate_condition("val", {"val": "0"}) is False

    def test_numeric_string_equality(self):
        """exitCode is stored as int, compared to literal 0."""
        assert _evaluate_condition("exitCode == 0", {"exitCode": "0"}) is True


# ------------------------------------------------------------------
# Branch expressions in the walker
# ------------------------------------------------------------------

class TestBranchExpressionsInWalker:
    """Test expression-based branches running through GraphWalker."""

    @staticmethod
    def _make_branch_flowchart(condition: str) -> Flowchart:
        return Flowchart(
            blocks={
                "s": StartBlock(id="s"),
                "b": BranchBlock(id="b", name="Check", condition=condition),
                "ok": EndBlock(id="ok", name="OK"),
                "fail": EndBlock(id="fail", name="Fail"),
            },
            connections=[
                Connection(source_id="s", target_id="b"),
                Connection(source_id="b", target_id="ok", is_true_path=True),
                Connection(source_id="b", target_id="fail", is_true_path=False),
            ],
        )

    async def test_equality_branch_true(self):
        fc = self._make_branch_flowchart("exitCode == 0")
        walker = GraphWalker(fc, MockSession(), {"exitCode": 0}, MockProtocol())
        result = await walker.run()
        names = [e.block_name for e in result.log]
        assert "OK" in names
        assert "Fail" not in names

    async def test_equality_branch_false(self):
        fc = self._make_branch_flowchart("exitCode == 0")
        walker = GraphWalker(fc, MockSession(), {"exitCode": 1}, MockProtocol())
        result = await walker.run()
        names = [e.block_name for e in result.log]
        assert "Fail" in names
        assert "OK" not in names

    async def test_less_than_branch(self):
        fc = self._make_branch_flowchart("i < 3")
        walker = GraphWalker(fc, MockSession(), {"i": 1}, MockProtocol())
        result = await walker.run()
        names = [e.block_name for e in result.log]
        assert "OK" in names

    async def test_template_in_condition(self):
        """Condition with {{var}} template — like Pride's 'i < {{max}}'."""
        fc = self._make_branch_flowchart("i < {{max}}")
        walker = GraphWalker(fc, MockSession(), {"i": 1, "max": 5}, MockProtocol())
        result = await walker.run()
        names = [e.block_name for e in result.log]
        assert "OK" in names

    async def test_template_in_condition_false(self):
        fc = self._make_branch_flowchart("i < {{max}}")
        walker = GraphWalker(fc, MockSession(), {"i": 5, "max": 5}, MockProtocol())
        result = await walker.run()
        names = [e.block_name for e in result.log]
        assert "Fail" in names


# ------------------------------------------------------------------
# Variable type aliases (int/float -> number)
# ------------------------------------------------------------------

class TestVariableTypeAliases:
    async def test_int_type_in_variable_block(self):
        """Pride uses 'int' as variable_type; should map to NUMBER."""
        fc = Flowchart.model_validate({
            "blocks": {
                "s": {"type": "start", "id": "s"},
                "v": {
                    "type": "variable", "id": "v",
                    "variable_name": "i",
                    "variable_value": "0",
                    "variable_type": "int",
                },
                "e": {"type": "end", "id": "e"},
            },
            "connections": [
                {"source_id": "s", "target_id": "v"},
                {"source_id": "v", "target_id": "e"},
            ],
        })
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())
        result = await walker.run()
        assert result.status == "completed"
        assert result.variables["i"] == 0.0

    async def test_float_type_in_bash_output(self):
        """Pride uses 'int' as output_type; should map to NUMBER."""
        fc = Flowchart.model_validate({
            "blocks": {
                "s": {"type": "start", "id": "s"},
                "b": {
                    "type": "bash", "id": "b",
                    "command": "echo 42",
                    "capture_output": True,
                    "output_variable": "result",
                    "output_type": "int",
                },
                "e": {"type": "end", "id": "e"},
            },
            "connections": [
                {"source_id": "s", "target_id": "b"},
                {"source_id": "b", "target_id": "e"},
            ],
        })
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())
        result = await walker.run()
        assert result.status == "completed"
        assert result.variables["result"] == 42.0


# ------------------------------------------------------------------
# Pride-format JSON parsing (source_block_id, sound_effect, etc.)
# ------------------------------------------------------------------

class TestPrideJsonParsing:
    def test_connection_with_source_block_id(self):
        """Pride uses source_block_id/target_block_id."""
        c = Connection.model_validate({
            "id": "abc",
            "source_block_id": "src",
            "target_block_id": "tgt",
            "source_port": "bottom",
            "target_port": "top",
            "is_true_path": True,
            "condition": None,
            "label": None,
        })
        assert c.source_id == "src"
        assert c.target_id == "tgt"

    def test_connection_axi_format_still_works(self):
        c = Connection.model_validate({
            "source_id": "a",
            "target_id": "b",
        })
        assert c.source_id == "a"
        assert c.target_id == "b"

    def test_connection_serializes_canonical(self):
        """Round-trip should use source_id/target_id, not Pride names."""
        c = Connection.model_validate({
            "source_block_id": "a",
            "target_block_id": "b",
        })
        d = c.model_dump()
        assert "source_id" in d
        assert "source_block_id" not in d

    def test_prompt_block_ignores_sound_effect(self):
        """Pride PromptBlocks have sound_effect — should be silently ignored."""
        fc = Flowchart.model_validate({
            "blocks": {
                "s": {"type": "start", "id": "s"},
                "p": {
                    "type": "prompt", "id": "p",
                    "prompt": "Hello",
                    "output_schema": None,
                    "sound_effect": "ding.wav",
                },
                "e": {"type": "end", "id": "e"},
            },
            "connections": [
                {"source_id": "s", "target_id": "p"},
                {"source_id": "p", "target_id": "e"},
            ],
        })
        assert len(fc.blocks) == 3

    def test_flowchart_ignores_start_block_id(self):
        """Pride flowcharts have start_block_id — should be ignored."""
        fc = Flowchart.model_validate({
            "blocks": {
                "s": {"type": "start", "id": "s"},
                "e": {"type": "end", "id": "e"},
            },
            "connections": [{"source_id": "s", "target_id": "e"}],
            "start_block_id": "s",
        })
        assert len(fc.blocks) == 2


# ------------------------------------------------------------------
# Full Pride flowchart loading
# ------------------------------------------------------------------

class TestPrideExamplesLoad:
    """Load every Pride example and verify it parses."""

    @pytest.mark.parametrize("name", [
        "ex0-design-doc",
        "ex1-do-until-done",
        "ex2-testing-loop",
        "ex3-improve-project",
        "all-examples",
    ])
    def test_load_pride_command(self, name: str):
        path = PRIDE_COMMANDS_DIR / f"{name}.json"
        cmd = load_command(path)
        assert cmd.name == name
        assert len(cmd.flowchart.blocks) > 0
        assert len(cmd.flowchart.connections) > 0


# ------------------------------------------------------------------
# End-to-end: Pride-format for-loop flowchart
# ------------------------------------------------------------------

class TestPrideForLoop:
    """Simulates the pattern from ex3-improve-project:
    variable(int) -> branch(expression) -> bash(increment) -> loop back.
    """

    async def test_pride_style_for_loop(self):
        """A for-loop using Pride conventions: int types, expression branch, bash increment."""
        fc = Flowchart.model_validate({
            "blocks": {
                "s": {"type": "start", "id": "s", "name": "START",
                       "position": {"x": 0, "y": 0}},
                "set_max": {
                    "type": "variable", "id": "set_max", "name": "SET max",
                    "variable_name": "max", "variable_value": "3",
                    "variable_type": "int",
                    "position": {"x": 0, "y": 100},
                },
                "set_i": {
                    "type": "variable", "id": "set_i", "name": "SET i",
                    "variable_name": "i", "variable_value": "0",
                    "variable_type": "int",
                    "position": {"x": 0, "y": 200},
                },
                "check": {
                    "type": "branch", "id": "check", "name": "i<max",
                    "condition": "i < {{max}}",
                    "position": {"x": 0, "y": 300},
                },
                "inc": {
                    "type": "bash", "id": "inc", "name": "i++",
                    "command": "echo $(({{i}}+1))",
                    "capture_output": True,
                    "output_variable": "i",
                    "output_type": "int",
                    "position": {"x": 100, "y": 300},
                },
                "e": {"type": "end", "id": "e", "name": "END",
                       "position": {"x": 0, "y": 400}},
            },
            "connections": [
                {"source_block_id": "s", "target_block_id": "set_max",
                 "source_port": "bottom", "target_port": "top", "is_true_path": True},
                {"source_block_id": "set_max", "target_block_id": "set_i",
                 "source_port": "bottom", "target_port": "top", "is_true_path": True},
                {"source_block_id": "set_i", "target_block_id": "check",
                 "source_port": "bottom", "target_port": "top", "is_true_path": True},
                {"source_block_id": "check", "target_block_id": "inc",
                 "source_port": "right", "target_port": "left", "is_true_path": True},
                {"source_block_id": "check", "target_block_id": "e",
                 "source_port": "bottom", "target_port": "top", "is_true_path": False},
                {"source_block_id": "inc", "target_block_id": "check",
                 "source_port": "bottom", "target_port": "top", "is_true_path": True},
            ],
            "start_block_id": "s",
        })
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())
        result = await walker.run()

        assert result.status == "completed"
        # Should have looped 3 times (i: 0->1->2->3, then 3 < 3 is false)
        assert result.variables["i"] == 3.0
        assert result.variables["max"] == 3.0

    async def test_pride_exit_code_branch(self):
        """Simulates ex2-testing-loop pattern: bash sets exitCode, branch checks == 0."""
        fc = Flowchart.model_validate({
            "blocks": {
                "s": {"type": "start", "id": "s"},
                "run": {
                    "type": "bash", "id": "run",
                    "command": "exit 0",
                    "continue_on_error": True,
                    "exit_code_variable": "exitCode",
                },
                "check": {
                    "type": "branch", "id": "check", "name": "TESTS PASSED?",
                    "condition": "exitCode == 0",
                },
                "pass": {"type": "end", "id": "pass", "name": "PASS"},
                "fail": {"type": "end", "id": "fail", "name": "FAIL"},
            },
            "connections": [
                {"source_block_id": "s", "target_block_id": "run",
                 "is_true_path": True},
                {"source_block_id": "run", "target_block_id": "check",
                 "is_true_path": True},
                {"source_block_id": "check", "target_block_id": "pass",
                 "is_true_path": True},
                {"source_block_id": "check", "target_block_id": "fail",
                 "is_true_path": False},
            ],
        })
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())
        result = await walker.run()
        names = [e.block_name for e in result.log]
        assert "PASS" in names
        assert "FAIL" not in names

    async def test_pride_exit_code_branch_failure(self):
        """Same pattern but command exits non-zero."""
        fc = Flowchart.model_validate({
            "blocks": {
                "s": {"type": "start", "id": "s"},
                "run": {
                    "type": "bash", "id": "run",
                    "command": "exit 1",
                    "continue_on_error": True,
                    "exit_code_variable": "exitCode",
                },
                "check": {
                    "type": "branch", "id": "check", "name": "TESTS PASSED?",
                    "condition": "exitCode == 0",
                },
                "pass": {"type": "end", "id": "pass", "name": "PASS"},
                "fail": {"type": "end", "id": "fail", "name": "FAIL"},
            },
            "connections": [
                {"source_block_id": "s", "target_block_id": "run",
                 "is_true_path": True},
                {"source_block_id": "run", "target_block_id": "check",
                 "is_true_path": True},
                {"source_block_id": "check", "target_block_id": "pass",
                 "is_true_path": True},
                {"source_block_id": "check", "target_block_id": "fail",
                 "is_true_path": False},
            ],
        })
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())
        result = await walker.run()
        names = [e.block_name for e in result.log]
        assert "FAIL" in names
        assert "PASS" not in names
