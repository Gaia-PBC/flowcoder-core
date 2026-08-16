"""Gating for the local-model tier.

These tests drive a real, locally served model rather than the stub, so they
only mean anything when one is actually running.  They are *deselected* when it
is not — never skipped.  The repo's policy hook turns a skip into a failure on
the grounds that a skipped test is evidence of nothing, and deselection is the
sanctioned way out: deselected tests are never collected and produce no report.

Set two environment variables to run them::

    FLOWCODER_LOCAL_MODEL=my-served-model-name
    ANTHROPIC_BASE_URL=http://localhost:8000

The second is not optional once the first is set.  Without it the `claude` CLI
falls back to the real Anthropic API, and a testbench sweep meant to be free
would quietly bill the account instead — so that combination is a hard error
rather than a silent redirect.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

LOCAL_MODEL_ENV = "FLOWCODER_LOCAL_MODEL"
BASE_URL_ENV = "ANTHROPIC_BASE_URL"
TIMEOUT_ENV = "FLOWCODER_LOCAL_MODEL_TIMEOUT"


def local_model() -> str | None:
    return os.environ.get(LOCAL_MODEL_ENV) or None


def base_url() -> str | None:
    return os.environ.get(BASE_URL_ENV) or None


def run_timeout() -> float:
    """Per-run bound.  Local models are slower than the API, often much."""
    return float(os.environ.get(TIMEOUT_ENV, "300"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Drop this directory's tests when there is no local model to talk to.

    Only items under this conftest's own directory are touched — the hook is
    handed every collected item in the session, including the stub-based tiers
    that must keep running.
    """
    if local_model():
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
def model_name() -> str:
    """The served model name, with the misconfiguration guard applied once."""
    name = local_model()
    assert name, "collection should have deselected these tests"
    if not base_url():
        raise pytest.UsageError(
            f"{LOCAL_MODEL_ENV}={name} is set but {BASE_URL_ENV} is not. "
            "The claude CLI would fall back to the real Anthropic API and bill "
            f"the account. Set {BASE_URL_ENV} to the local endpoint "
            "(e.g. http://localhost:8000, with no trailing /v1)."
        )
    return name
