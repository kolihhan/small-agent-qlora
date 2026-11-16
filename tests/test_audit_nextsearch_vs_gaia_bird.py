from scripts.audit_nextsearch_vs_gaia_bird import norm, similarity,extract_reference
import pytest

def test_task_lexical_exact_and_near():
    a="In which year did the artist from the capital of a country publish the novel?"
    b="In which year did the artist from the capital of a country publish the novel!"
    assert norm(a)==norm(b)
    assert similarity(a,b)[1]>=0.95

def test_references_fail_closed_on_duplicate_ids():
    rows=[{"task_id":"x","Question":"Which year did the book appear?"}]*2
    with pytest.raises(ValueError,match="duplicate"):
        extract_reference(rows,"gaia-test",2)

def test_input_not_trivially_normalized_away():
    assert norm("Which country?") != norm("Which countries?")

def test_different_questions_are_not_near():
    a="What is the author of the green movie about a river?"
    b="How many clubs have scored more than twelve football goals?"
    lexical,char,ratio=similarity(a,b)
    assert max(lexical,char,ratio)<0.72
