def _summary(rate, *, stop_reasons=None):
    return {
        "total": 200,
        "correct": int(rate * 2),
        "exact_success_rate": float(rate),
        "completed": 200,
        "completion_rate": 100.0,
        "stop_reasons": stop_reasons or {"final": 200},
    }


def test_policy_promotion_passes_at_exactly_ten_percentage_points():
    from gaia_small_agent.training.policy_promotion import evaluate_policy_promotion

    verdict = evaluate_policy_promotion(_summary(50.0), _summary(60.0))

    assert verdict["promote"] is True
    assert verdict["exact_pp_delta"] == 10.0
    assert verdict["reasons"] == []


def test_policy_promotion_rejects_below_ten_percentage_points():
    from gaia_small_agent.training.policy_promotion import evaluate_policy_promotion

    verdict = evaluate_policy_promotion(_summary(50.0), _summary(59.9))

    assert verdict["promote"] is False
    assert verdict["exact_pp_delta"] == 9.9
    assert "exact_success_gain_below_10.0pp" in verdict["reasons"]


def test_policy_promotion_rejects_increased_catastrophic_failures():
    from gaia_small_agent.training.policy_promotion import evaluate_policy_promotion

    baseline = _summary(50.0, stop_reasons={"final": 199, "model_timeout": 1})
    candidate = _summary(65.0, stop_reasons={"final": 198, "model_timeout": 2})
    verdict = evaluate_policy_promotion(baseline, candidate)

    assert verdict["promote"] is False
    assert verdict["exact_pp_delta"] == 15.0
    assert verdict["catastrophic_regressions"] == {"model_timeout": 1}
    assert "catastrophic_failure_regression" in verdict["reasons"]


def test_non_catastrophic_stop_reason_does_not_block_promotion():
    from gaia_small_agent.training.policy_promotion import evaluate_policy_promotion

    baseline = _summary(40.0, stop_reasons={"final": 180, "max_steps": 20})
    candidate = _summary(55.0, stop_reasons={"final": 175, "max_steps": 25})
    verdict = evaluate_policy_promotion(baseline, candidate)

    assert verdict["promote"] is True
    assert verdict["catastrophic_regressions"] == {}
