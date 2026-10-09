from __future__ import annotations


CATASTROPHIC_STOP_REASONS = frozenset(
    {"model_capacity", "model_timeout", "model_unavailable", "model_error"}
)


def evaluate_shadow_promotion(base: dict, candidate: dict) -> dict:
    """Decide promotion from aggregate blind-Shadow metrics only."""
    correct_delta = int(candidate.get("correct", 0)) - int(base.get("correct", 0))
    completion_delta = int(candidate.get("completed", 0)) - int(base.get("completed", 0))

    base_stops = dict(base.get("stop_reasons") or {})
    candidate_stops = dict(candidate.get("stop_reasons") or {})
    new_catastrophic_failures = sorted(
        reason
        for reason in CATASTROPHIC_STOP_REASONS
        if int(candidate_stops.get(reason, 0)) > 0 and int(base_stops.get(reason, 0)) == 0
    )

    reasons: list[str] = []
    if correct_delta < 2:
        reasons.append("correct_delta_below_2")
    if completion_delta < -1:
        reasons.append("completion_regressed_by_more_than_1")
    if new_catastrophic_failures:
        reasons.append("new_catastrophic_runtime_failure")

    return {
        "promote": not reasons,
        "correct_delta": correct_delta,
        "completion_delta": completion_delta,
        "new_catastrophic_failures": new_catastrophic_failures,
        "reasons": reasons,
    }
