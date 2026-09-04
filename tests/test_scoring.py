from gaia_small_agent.benchmark.scoring import gaia_score


def test_gaia_string_normalization_removes_all_whitespace():
    assert gaia_score("sea gull", "seagull") is True


def test_gaia_comma_ground_truth_is_a_list_not_a_number():
    assert gaia_score("1,2", "1,2") is True
    assert gaia_score("12", "1,2") is False


def test_gaia_numeric_answer_strips_common_units_from_model_answer():
    assert gaia_score("$1,234", "1234") is True
