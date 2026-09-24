"""Evaluation harness (Phase 6): metrics, tiered sampling, MLflow logging.

Kept separate from the shipped pipeline package: nothing in `transcript_task`
proper imports from here, so a user who only wants the transcription tool
never pulls in jiwer/sentence-transformers/mlflow. Only `scripts/benchmark.py`
imports both sides.
"""

from __future__ import annotations
