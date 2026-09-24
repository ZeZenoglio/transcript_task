"""Re-exports `normalize_pt` from `transcript_task.text_compare`.

Moved there in Phase 6's runtime-guard follow-up so the production pipeline
(`pipeline.py`'s refine-quality guard) can reuse the same normalizer without
importing anything from `eval/` -- this module's siblings pull in jiwer/
sentence-transformers/mlflow, none of which the shipped pipeline needs. This
module stays as a thin alias so existing `eval.normalizer` imports keep
working unchanged.
"""

from __future__ import annotations

from ..text_compare import normalize_pt

__all__ = ["normalize_pt"]
