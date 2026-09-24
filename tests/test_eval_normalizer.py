from transcript_task.eval.normalizer import normalize_pt


def test_casefolds():
    assert normalize_pt("CAFÉ") == normalize_pt("café")


def test_strips_punctuation():
    assert normalize_pt('"Correntes de retorno", disse ele.') == "correntes de retorno disse ele"


def test_expands_digits_to_pt_words():
    assert normalize_pt("Ele tem 12 anos") == "ele tem doze anos"


def test_collapses_whitespace():
    assert normalize_pt("um   dois\ttres\n\nquatro") == "um dois tres quatro"


def test_idempotent():
    text = 'Ele tem 12 anos, "certo"?'
    once = normalize_pt(text)
    assert normalize_pt(once) == once


def test_accents_kept_by_default():
    assert normalize_pt("más") != normalize_pt("mas")


def test_strip_accents_option_folds_them():
    assert normalize_pt("más", strip_accents=True) == normalize_pt("mas", strip_accents=True)


def test_empty_string():
    assert normalize_pt("") == ""


def test_huge_digit_run_does_not_crash():
    # num2words has real limits; falling back to leaving it as digits (and
    # taking the WER hit) beats crashing the whole benchmark run over one
    # pathological clip.
    huge = "9" * 200
    assert normalize_pt(huge) == huge
