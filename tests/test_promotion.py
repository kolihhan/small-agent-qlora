from gaia_small_agent.benchmark.promotion import evaluate_shadow_promotion


def _summary(*, correct, completed=25, stop_reasons=None, tool_calls=100):
    return {
        "correct": correct,
        "completed": completed,
        "total": 25,
        "tool_calls": tool_calls,
        "stop_reasons": stop_reasons or {"final": completed},
    }


def test_shadow_promotion_requires_two_more_correct_without_completion_regression():
    base = _summary(correct=4, completed=23)
    candidate = _summary(correct=6, completed=22)

    result = evaluate_shadow_promotion(base, candidate)

    assert result["promote"] is True
    assert result["correct_delta"] == 2
    assert result["completion_delta"] == -1
    assert result["new_catastrophic_failures"] == []
    assert result["reasons"] == []


def test_shadow_promotion_rejects_only_one_more_correct():
    result = evaluate_shadow_promotion(_summary(correct=4), _summary(correct=5))

    assert result["promote"] is False
    assert "correct_delta_below_2" in result["reasons"]


def test_shadow_promotion_rejects_efficiency_only_improvement():
    result = evaluate_shadow_promotion(
        _summary(correct=4, tool_calls=100),
        _summary(correct=4, tool_calls=20),
    )

    assert result["promote"] is False
    assert result["correct_delta"] == 0


def test_shadow_promotion_rejects_two_task_completion_regression():
    result = evaluate_shadow_promotion(
        _summary(correct=4, completed=24),
        _summary(correct=6, completed=22),
    )

    assert result["promote"] is False
    assert result["completion_delta"] == -2
    assert "completion_regressed_by_more_than_1" in result["reasons"]


def test_shadow_promotion_rejects_new_catastrophic_runtime_failure():
    base = _summary(correct=4, stop_reasons={"final": 25})
    candidate = _summary(correct=6, stop_reasons={"final": 24, "model_capacity": 1})

    result = evaluate_shadow_promotion(base, candidate)

    assert result["promote"] is False
    assert result["new_catastrophic_failures"] == ["model_capacity"]
    assert "new_catastrophic_runtime_failure" in result["reasons"]


def test_existing_catastrophic_reason_does_not_count_as_new():
    base = _summary(correct=4, stop_reasons={"final": 24, "model_timeout": 1})
    candidate = _summary(correct=6, stop_reasons={"final": 23, "model_timeout": 2})

    result = evaluate_shadow_promotion(base, candidate)

    assert result["new_catastrophic_failures"] == []
    assert result["promote"] is True
