"""Shared fixtures / policy hooks for flowcoder-flowchart tests."""

import pytest


# ── Policy: skips are failures (SOUL prompting fix #5) ────────────────
# A skipped/xfailed test produces zero evidence, so a suite may not
# contain one and still be reported as passing. Convert any skip/xfail
# outcome into a failure. Deselection (e.g. -m "not slow") is untouched:
# deselected tests are never collected and produce no report.
@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.skipped:
        report.outcome = "failed"
        report.longrepr = f"SKIP DISALLOWED by project policy (skips are failures): {report.longrepr}"
    return report
