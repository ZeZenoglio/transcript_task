"""Layering check for transcript_task.text_compare: this module backs a
production runtime guard (pipeline.py's stage_refine), so it must be
importable -- and its functions must work -- without touching anything in
transcript_task.eval (which pulls in jiwer/sentence-transformers/mlflow).
Behavioral coverage for normalize_pt/content_recall/length_ratio lives in
tests/test_eval_normalizer.py and tests/test_eval_metrics.py, which exercise
the same functions via eval's re-exports.
"""

import sys

import pytest


def test_importable_without_the_eval_package_loaded():
    for name in list(sys.modules):
        if name == "transcript_task.eval" or name.startswith("transcript_task.eval."):
            pytest.skip("eval package already imported by another test in this run")
    from transcript_task import text_compare  # noqa: F401

    assert "transcript_task.eval" not in sys.modules


def test_core_functions_work_standalone():
    from transcript_task.text_compare import content_recall, length_ratio, normalize_pt

    # num2words doesn't do Portuguese gender agreement ("dois"/"duas"), so
    # this is genuinely what normalize_pt produces -- not the grammatically
    # "correct" duas vezes. A known, minor normalizer limitation; see
    # text_compare.py.
    assert normalize_pt("Café, 2 vezes!") == "café dois vezes"
    assert content_recall("o gato dorme", "o gato dorme calmamente") == 1.0
    assert length_ratio("abcde", "abcde") == 1.0
