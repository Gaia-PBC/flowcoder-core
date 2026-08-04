"""Shared fixtures / policy hooks for the (engine-only) root test tree.

Kept after the GUI/engine repo separation so the no-skips policy hook still
covers the surviving root tests.  It also re-exports MockSession/MockProtocol
from the engine test conftest: the surviving spawn_wait tests do
``from conftest import MockSession`` and this root conftest would otherwise
shadow the engine one on ``sys.modules['conftest']``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-export the engine test mocks.  Loaded by file path (not by the name
# "conftest") to avoid clashing with this module, which pytest imports as
# ``conftest``.
_engine_conftest = (
    Path(__file__).parent.parent / "packages" / "flowcoder-engine" / "tests" / "conftest.py"
)
_spec = importlib.util.spec_from_file_location("flowcoder_engine_test_conftest", _engine_conftest)
_engine_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine_mod)
MockSession = _engine_mod.MockSession
MockProtocol = _engine_mod.MockProtocol


# ── Policy: skips are failures (SOUL prompting fix #5) ────────────────
# A skipped/xfailed test produces zero evidence, so a suite may not contain
# one and still be reported as passing.  Convert any skip/xfail outcome into a
# failure.  Deselection (e.g. -m "not slow") is untouched: deselected tests are
# never collected and produce no report.
@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.skipped:
        report.outcome = "failed"
        report.longrepr = f"SKIP DISALLOWED by project policy (skips are failures): {report.longrepr}"
    return report
