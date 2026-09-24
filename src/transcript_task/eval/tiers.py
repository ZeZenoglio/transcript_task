"""Tiered benchmark sampling (PLAN.md Phase 6).

`smoke`  -- the handful of fixtures already committed to the repo (tests/
            fixtures/manifest.jsonl), for CI and unit tests with no network.
`quick`  -- a fixed-seed, duration-stratified random sample of the full
            manifest, small enough to run in a few minutes.
`full`   -- everything.

Stratifying by duration matters because a plain random sample of FLEURS
skews toward short clips (most of the split is short read sentences); a
`quick` tier that's all 3-6s clips would never exercise the long-form
repetition-loop failure mode Phase 2 found. Bucketing by duration quantile
first, then sampling proportionally from each bucket, keeps the subsample's
duration spread representative of the full split.
"""

from __future__ import annotations

import random

DEFAULT_QUICK_N = 30
DEFAULT_SEED = 42
DEFAULT_BUCKETS = 5


def _stratified_sample(rows: list[dict], n: int, seed: int, buckets: int) -> list[dict]:
    if n >= len(rows):
        return list(rows)

    ordered = sorted(rows, key=lambda r: r["duration"])
    bucket_count = min(buckets, n, len(ordered)) or 1
    bucket_size = len(ordered) / bucket_count
    rng = random.Random(seed)

    # Split into `bucket_count` contiguous duration-sorted slices, then take
    # a proportional, independently-shuffled share from each so the sample's
    # duration histogram mirrors the full split's rather than collapsing to
    # one narrow band.
    picked: list[dict] = []
    remaining_n = n
    for i in range(bucket_count):
        start = round(i * bucket_size)
        end = round((i + 1) * bucket_size) if i < bucket_count - 1 else len(ordered)
        bucket = ordered[start:end]
        buckets_left = bucket_count - i
        take = max(1, round(remaining_n / buckets_left)) if bucket else 0
        take = min(take, len(bucket), remaining_n)
        picked.extend(rng.sample(bucket, take))
        remaining_n -= take

    # Rounding can leave the sample a clip or two short of `n`; top up from
    # whatever wasn't already picked, oldest-bucket-first, deterministically.
    if remaining_n > 0:
        leftovers = [r for r in ordered if r not in picked]
        picked.extend(rng.sample(leftovers, min(remaining_n, len(leftovers))))

    return picked


def select_tier(
    manifest: list[dict],
    smoke_manifest: list[dict],
    tier: str,
    *,
    n: int = DEFAULT_QUICK_N,
    seed: int = DEFAULT_SEED,
    buckets: int = DEFAULT_BUCKETS,
) -> list[dict]:
    """Pick the clips for one benchmark tier.

    `manifest` is the full FLEURS manifest (data/fleurs_pt/manifest.jsonl,
    fetched separately); `smoke_manifest` is the small committed one
    (tests/fixtures/manifest.jsonl) so `smoke` never needs the full download.
    """
    if tier == "smoke":
        return list(smoke_manifest)
    if tier == "full":
        return list(manifest)
    if tier == "quick":
        return _stratified_sample(manifest, n, seed, buckets)
    raise ValueError(f"unknown tier: {tier!r} (expected smoke, quick, or full)")
