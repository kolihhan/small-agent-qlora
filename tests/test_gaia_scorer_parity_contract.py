from scripts.check_gaia_scorer_parity import synthetic_cases
from gaia_small_agent.benchmark.scoring import gaia_score

def test_synthetic_scorer_input_suite_has_expected_edge_cases():
    cases=synthetic_cases()
    assert len(cases)>=100
    assert len(set(str(row) for row in cases))>=50
    assert gaia_score("42%","42") is True
    assert gaia_score("The answer is 42","42") is False
    assert gaia_score("New York","new york") is True
    assert gaia_score("A; B","a,b") is True
