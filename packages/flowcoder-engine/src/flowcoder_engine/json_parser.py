"""Extract JSON from Claude's text responses.

Claude may wrap JSON in markdown code blocks or include extra text, and it
often reasons *before* emitting the answer -- quoting brace-containing code or
even scratch JSON in a fenced block. This module extracts the intended JSON
object, preferring the **last** valid object in the text (the answer comes
last), so reasoning that precedes the answer can no longer shadow it.
"""

from __future__ import annotations

import json
import re
from typing import Any

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def parse_json_from_response(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from a response string.

    Tries in order:
    1. The entire text as JSON.
    2. The **last** parseable object among all ``` fenced blocks.
    3. The **last** parseable ``{ ... }`` object found by a balanced-brace scan.
    4. Legacy fallback: the greedy first-``{`` to last-``}`` span.

    Preferring the *last* object in (2) and (3) makes the parser robust to a
    model that emits scratch JSON or quotes brace-containing code in its
    reasoning before the answer -- the previous implementation grabbed the
    *first* fenced block (returning the wrong object) or let a greedy brace
    match span from the reasoning into the answer (returning None).

    Returns None if no valid JSON object is found.
    """
    # 1. The whole string.
    parsed = _try_parse(text.strip())
    if parsed is not None:
        return parsed

    # 2. Fenced code blocks -- prefer the LAST parseable, non-empty object.
    fenced = [m.group(1).strip() for m in _CODE_FENCE_RE.finditer(text)]
    parsed = _last_parseable(fenced)
    if parsed is not None:
        return parsed

    # 3. Balanced { ... } objects -- prefer the LAST parseable, non-empty one.
    parsed = _last_parseable(_iter_brace_objects(text))
    if parsed is not None:
        return parsed

    # 4. Legacy fallback: greedy first { to last } (unchanged safety net for
    #    inputs the scans above don't isolate, e.g. unbalanced quotes in prose).
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return _try_parse(brace_match.group(0))

    return None


def _last_parseable(candidates: list[str]) -> dict[str, Any] | None:
    """Return the last candidate that parses to a non-empty dict, else None."""
    for candidate in reversed(candidates):
        parsed = _try_parse(candidate)
        if parsed:  # non-empty dict
            return parsed
    return None


def _iter_brace_objects(text: str) -> list[str]:
    """Return top-level ``{ ... }`` substrings via a string-aware brace scan.

    Only double quotes delimit strings (as in JSON), so apostrophes in prose
    don't desync the scan, and braces that appear inside string literals are
    ignored. Returns objects in document order.
    """
    objects: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    objects.append(text[start : i + 1])
    return objects


def _try_parse(text: str) -> dict[str, Any] | None:
    """Try to parse text as JSON dict. Returns None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None
