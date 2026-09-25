"""Auto-marks every collected test by which tier directory it lives in, so
individual files under tests/unit/ and tests/eval/ don't need boilerplate
`pytestmark = pytest.mark.unit` lines, and `pytest -m <tier>` selection is
exact and can't drift from where a file happens to sit. tests/integration/
tests additionally carry an explicit `@pytest.mark.integration` decorator
(kept for clarity when a file is read on its own) -- applying the same
marker twice here is harmless.

tests/eval/ (metric-correctness against fakes, no real models) counts as
`unit` for marker purposes: it has the same fast/offline profile as
tests/unit/, just organised separately per PLAN.md's Phase 10 layout.
"""

from __future__ import annotations

from pathlib import Path

_TIER_BY_TOP_DIR = {
    "unit": "unit",
    "eval": "unit",
    "integration": "integration",
    "e2e": "e2e",
}

_TESTS_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        top_dir = Path(item.fspath).relative_to(_TESTS_ROOT).parts[0]
        tier = _TIER_BY_TOP_DIR.get(top_dir)
        if tier:
            item.add_marker(tier)
