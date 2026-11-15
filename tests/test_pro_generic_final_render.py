import pytest
from scripts.pro_generic_final_render import render_final_answer


@pytest.mark.parametrize("raw,want,reason", [
    ("Evidence line 1. Evidence line 2.\n\nAnswer: Ada Lovelace",
     "Ada Lovelace", "explicit_final_field"),
    ("Evidence is balanced.\n\nSecret Svensson",
     "Secret Svensson", "standalone_final_line"),
    ("After checking the data:\n\nFinal answer: 1986",
     "1986", "explicit_final_field"),
    ("The proof is inconclusive.\n\nI cannot answer from available evidence.",
     "The proof is inconclusive.\n\nI cannot answer from available evidence.",
     "no_safe_final_field"),
    ("The answer is not obvious.\nIt might be A or B.",
     "The answer is not obvious.\nIt might be A or B.", "no_safe_final_field"),
    ("Project X", "Project X", "unchanged_single_line"),
    ("Reason.\n\nAnswer: " + ("x" * 200),
     "Reason.\n\nAnswer: " + ("x" * 200), "ambiguous_explicit_field"),
    ("Facts.\n\nanswer: Blue Ridge", "Blue Ridge", "explicit_final_field"),
])
def test_renderer_only_extracts_explicit_or_separate_short_final(raw, want, reason):
    assert render_final_answer(raw) == (want, reason)


def test_plain_sentences_never_guess_reference_or_choice():
    raw = "Movie X is directed by Bob. Movie Y is directed by Alice. Bob died earlier."
    assert render_final_answer(raw)[0] == raw


def test_one_line_direct_answer_is_invariant():
    assert render_final_answer("March 2024")[0] == "March 2024"


def test_reject_nonstring_or_empty_without_hallucination():
    assert render_final_answer("   ") == ("", "unchanged_single_line")
    with pytest.raises(TypeError):
        render_final_answer(None)
