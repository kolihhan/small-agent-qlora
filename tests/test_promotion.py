from gaia_small_agent.benchmark.gaia100 import summarize_results


def _summary(*, correct=4, completed=24, stop_reasons=None, tool_calls=50):
    return {
        "correct": correct,
        "completed": completed,
        "tool_calls": tool_calls,
        "stop_reasons": dict(stop_reasons or {}),
    }


def test_shadow_promotion_passes_only_with_two_more_correct_and_stable_completion():
    from gaia_small_agent.benchmark.promotion import evaluate_shadow_promotion

    result = evaluate_shadow_promotion(
        _summary(correct=4, completed=24),
        _summary(correct=6, completed=23),
    )

    assert result["promote"] is True
    assert result["correct_delta"] == 2
    assert result["completion_delta"] == -1
    assert result["new_catastrophic_failures"] == []
    assert result["reasons"] == []


def test_shadow_promotion_rejects_only_one_more_correct():
    from gaia_small_agent.benchmark.promotion import evaluate_shadow_promotion

    result = evaluate_shadow_promotion(_summary(correct=4), _summary(correct=5))

    assert result["promote"] is False
    assert "correct_delta_below_2" in result["reasons"]


def test_shadow_promotion_rejects_efficiency_only_gain():
    from gaia_small_agent.benchmark.promotion import evaluate_shadow_promotion

    result = evaluate_shadow_promotion(
        _summary(correct=4, tool_calls=100),
        _summary(correct=4, tool_calls=10),
    )

    assert result["promote"] is False
    assert result["correct_delta"] == 0


def test_shadow_promotion_rejects_completion_regression_beyond_one():
    from gaia_small_agent.benchmark.promotion import evaluate_shadow_promotion

    result = evaluate_shadow_promotion(
        _summary(correct=4, completed=24),
        _summary(correct=6, completed=22),
    )

    assert result["promote"] is False
    assert "completion_delta_below_minus_1" in result["reasons"]


def test_shadow_promotion_rejects_new_catastrophic_runtime_failure():
    from gaia_small_agent.benchmark.promotion import evaluate_shadow_promotion

    result = evaluate_shadow_promotion(
        _summary(correct=4, stop_reasons={"final": 24, "model_timeout": 1}),
        _summary(correct=6, stop_reasons={"final": 24, "model_timeout": 1, "model_capacity": 1}),
    )

    assert result["promote"] is False
    assert result["new_catastrophic_failures"] == ["model_capacity"]
    assert "new_catastrophic_failure" in result["reasons"]


def test_summary_includes_aggregate_stop_reasons_without_case_content():
    rows = [
        {
            "level": 1,
            "correct": True,
            "completed": True,
            "stop_reason": "final",
            "tool_calls": 1,
            "tool_successes": 1,
            "tool_errors": 0,
            "steps": 1,
            "capability_gap": None,
            "failure_label": "correct",
        },
        {
            "level": 2,
            "correct": False,
            "completed": False,
            "stop_reason": "model_timeout",
            "tool_calls": 0,
            "tool_successes": 0,
            "tool_errors": 0,
            "steps": 1,
            "capability_gap": None,
            "failure_label": "incomplete_other",
        },
    ]

    summary = summarize_results(rows)

    assert summary["stop_reasons"] == {"final": 1, "model_timeout": 1}
