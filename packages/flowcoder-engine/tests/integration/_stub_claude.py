#!/usr/bin/env python3
"""A zero-token stub of the ``claude`` CLI for engine end-to-end tests.

The engine spawns its inner CLI as
``claude -p --input-format stream-json --output-format stream-json --verbose``.
This stub speaks the same JSON-lines protocol but costs nothing and needs no
network: it reads messages from stdin and, for every ``user`` message, emits an
``assistant`` text message followed by a terminal ``result`` message — which is
exactly what ``ClaudeSession.query`` reads until.  All CLI flags are ignored.

Run indirectly via ``engine_harness`` (it wires this in through ``--claude-path``).
Kept dependency-free (stdlib only) so any Python 3 interpreter can run it.
"""
from __future__ import annotations

import json
import sys


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    # readline() (not "for line in sys.stdin") so each line is surfaced as soon
    # as the engine flushes it, rather than waiting for a read-ahead buffer.
    while True:
        line = sys.stdin.readline()
        if not line:  # EOF — the engine closed our stdin, so shut down.
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg.get("type") == "user":
            content = msg.get("message", {}).get("content", "")
            reply = f"stub reply to: {content}"
            _emit(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": reply}],
                    },
                }
            )
            _emit(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": reply,
                    "num_turns": 1,
                    "duration_ms": 0,
                    "total_cost_usd": 0.0,
                    "session_id": "stub-session",
                }
            )
        # Any other message type (control_request, etc.) is ignored: a
        # start->input->end flowchart never elicits one from this stub.


if __name__ == "__main__":
    main()
