"""Tests for JSON extraction from text."""

from flowcoder_engine.json_parser import parse_json_from_response


class TestParseJsonFromResponse:
    def test_pure_json(self):
        result = parse_json_from_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```'
        result = parse_json_from_response(text)
        assert result == {"key": "value"}

    def test_json_in_generic_code_block(self):
        text = 'Result:\n```\n{"key": "value"}\n```'
        result = parse_json_from_response(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'The answer is {"valid": true, "reason": "looks good"} and that is it.'
        result = parse_json_from_response(text)
        assert result == {"valid": True, "reason": "looks good"}

    def test_no_json(self):
        result = parse_json_from_response("just plain text")
        assert result is None

    def test_json_array_ignored(self):
        # We only want dicts
        result = parse_json_from_response("[1, 2, 3]")
        assert result is None

    def test_nested_json(self):
        text = '{"outer": {"inner": true}}'
        result = parse_json_from_response(text)
        assert result == {"outer": {"inner": True}}

    def test_whitespace_json(self):
        text = """
        {
            "key": "value",
            "num": 42
        }
        """
        result = parse_json_from_response(text)
        assert result == {"key": "value", "num": 42}

    def test_empty_string(self):
        result = parse_json_from_response("")
        assert result is None

    def test_invalid_json(self):
        result = parse_json_from_response("{bad json}")
        assert result is None

    def test_json_fence_in_reasoning_before_answer(self):
        # A parseable JSON object emitted as scratch reasoning must not
        # shadow the real answer that comes later. Regression: the parser
        # grabbed the *first* fenced object -> wrong object / missing keys.
        text = (
            "Let me reason about this.\n"
            "Scratch object I considered:\n"
            "```json\n"
            '{"draft": true}\n'
            "```\n"
            "Final answer:\n"
            "```json\n"
            '{"correctness_score": 1, "reason": "ok"}\n'
            "```"
        )
        result = parse_json_from_response(text)
        assert result == {"correctness_score": 1, "reason": "ok"}

    def test_code_fence_with_braces_before_answer_fence(self):
        # A judge quoting brace-containing code in its reasoning must not
        # break extraction of the fenced answer. Regression: the first fence
        # failed to parse and the greedy brace fallback over-spanned -> None.
        text = (
            "Consider this code:\n"
            "```python\n"
            'd = {"x": 1}\n'
            "```\n"
            "Answer:\n"
            "```json\n"
            '{"correctness_score": 1}\n'
            "```"
        )
        result = parse_json_from_response(text)
        assert result == {"correctness_score": 1}

    def test_code_fence_with_braces_before_bare_answer(self):
        # Same reasoning shape, but the answer is a bare (unfenced) object:
        # the balanced-brace scan must return the last object rather than a
        # greedy span across the reasoning.
        text = (
            "Consider this code:\n"
            "```python\n"
            'd = {"x": 1}\n'
            "```\n"
            'Final: {"correctness_score": 1}'
        )
        result = parse_json_from_response(text)
        assert result == {"correctness_score": 1}

    def test_multiple_json_fences_returns_last(self):
        # When several parseable JSON fences are present, the last one is the
        # answer (reasoning/scratch comes first).
        text = (
            "```json\n"
            '{"step": 1}\n'
            "```\n"
            "```json\n"
            '{"step": 2, "final": true}\n'
            "```"
        )
        result = parse_json_from_response(text)
        assert result == {"step": 2, "final": True}
