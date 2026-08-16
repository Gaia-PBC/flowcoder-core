"""Gating for the local-model tier.

These tests drive a real, locally served model rather than the stub, so they
only mean anything when one is actually running.  They are *deselected* when it
is not — never skipped.  The repo's policy hook turns a skip into a failure on
the grounds that a skipped test is evidence of nothing, and deselection is the
sanctioned way out: deselected tests are never collected and produce no report.

The gate is ``ANTHROPIC_BASE_URL``, the same variable that points the `claude`
CLI at the endpoint in the first place::

    ANTHROPIC_BASE_URL=http://localhost:8000 uv run pytest .../local_model

Gating on the endpoint rather than on a flag of our own means these tests cannot
run against the billed API by construction — no endpoint, no tests — so there is
no misconfiguration left to guard against.  The trade is that a base URL
exported for some unrelated proxy also selects them; that is the intended
reading, since whatever it points at is what a flowchart would talk to.

``ANTHROPIC_MODEL`` names the served model if it is set.  When it is not, the
CLI's own default applies and no --model flag is passed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

BASE_URL_ENV = "ANTHROPIC_BASE_URL"
MODEL_ENV = "ANTHROPIC_MODEL"
TIMEOUT_ENV = "FLOWCODER_LOCAL_MODEL_TIMEOUT"


def base_url() -> str | None:
    return os.environ.get(BASE_URL_ENV) or None


def run_timeout() -> float:
    """Per-run bound.  Local models are slower than the API, often much."""
    return float(os.environ.get(TIMEOUT_ENV, "300"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Drop this directory's tests when there is no local endpoint to talk to.

    Only items under this conftest's own directory are touched — the hook is
    handed every collected item in the session, including the stub-based tiers
    that must keep running.
    """
    if base_url():
        return

    mine, theirs = [], []
    for item in items:
        (mine if HERE in Path(str(item.path)).parents or Path(str(item.path)).parent == HERE
         else theirs).append(item)
    if not mine:
        return

    items[:] = theirs
    config.hook.pytest_deselected(items=mine)


@pytest.fixture(scope="session")
def model_name() -> str | None:
    """The served model name, or None to let the CLI pick its default."""
    return os.environ.get(MODEL_ENV) or None
