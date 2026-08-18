from ControlSHINE.controlshine.metrics import answer_match, recoverability_label


def test_answer_match_observes_word_boundaries_and_aliases():
    assert answer_match("The answer is New York.", "New York")
    assert answer_match("NYC", "New York", ["NYC"])
    assert not answer_match("Yorkshire", "York")


def test_recoverability_label():
    assert recoverability_label(True, True, True) == "fully_recoverable"
    assert recoverability_label(True, False, True) == "unrecoverable_memory"
