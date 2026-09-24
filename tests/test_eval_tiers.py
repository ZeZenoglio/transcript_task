import pytest

from transcript_task.eval.tiers import select_tier

SMOKE = [{"audio_path": f"smoke{i}", "duration": 5.0} for i in range(4)]


def _synthetic_manifest(n: int = 100) -> list[dict]:
    # Duration spread from 1s to 100s, so bucketing is easy to verify.
    return [{"audio_path": f"clip{i}", "duration": float(i + 1)} for i in range(n)]


class TestSmokeTier:
    def test_returns_the_smoke_manifest_unchanged(self):
        full = _synthetic_manifest()
        result = select_tier(full, SMOKE, "smoke")
        assert result == SMOKE

    def test_ignores_n_and_seed(self):
        result = select_tier(_synthetic_manifest(), SMOKE, "smoke", n=1, seed=999)
        assert result == SMOKE


class TestFullTier:
    def test_returns_everything(self):
        full = _synthetic_manifest(50)
        assert select_tier(full, SMOKE, "full") == full


class TestQuickTier:
    def test_returns_requested_count(self):
        full = _synthetic_manifest(100)
        result = select_tier(full, SMOKE, "quick", n=30, seed=1)
        assert len(result) == 30

    def test_deterministic_for_the_same_seed(self):
        full = _synthetic_manifest(100)
        a = select_tier(full, SMOKE, "quick", n=30, seed=7)
        b = select_tier(full, SMOKE, "quick", n=30, seed=7)
        assert a == b

    def test_different_seeds_can_pick_different_clips(self):
        full = _synthetic_manifest(100)
        a = select_tier(full, SMOKE, "quick", n=30, seed=1)
        b = select_tier(full, SMOKE, "quick", n=30, seed=2)
        assert {r["audio_path"] for r in a} != {r["audio_path"] for r in b}

    def test_no_duplicate_clips_within_one_sample(self):
        full = _synthetic_manifest(100)
        result = select_tier(full, SMOKE, "quick", n=30, seed=3)
        paths = [r["audio_path"] for r in result]
        assert len(paths) == len(set(paths))

    def test_sample_spans_the_full_duration_range_not_just_short_clips(self):
        # Duration 1..100; a plain random sample of a real FLEURS-shaped
        # split (skewed toward short clips) could easily miss the long tail.
        # Stratification must not: some picked clip should come from each
        # half of the duration range.
        full = _synthetic_manifest(100)
        result = select_tier(full, SMOKE, "quick", n=30, seed=5)
        durations = [r["duration"] for r in result]
        assert min(durations) <= 20
        assert max(durations) >= 80

    def test_n_larger_than_manifest_returns_everything(self):
        full = _synthetic_manifest(10)
        result = select_tier(full, SMOKE, "quick", n=30, seed=1)
        assert len(result) == 10

    def test_small_manifest_with_small_n(self):
        full = _synthetic_manifest(5)
        result = select_tier(full, SMOKE, "quick", n=3, seed=1)
        assert len(result) == 3


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        select_tier(_synthetic_manifest(), SMOKE, "bogus")
