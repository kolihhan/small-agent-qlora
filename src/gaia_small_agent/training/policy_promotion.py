from __future__ import annotations

_CATASTROPHIC = ("model_capacity", "model_timeout", "model_unavailable", "model_error")


def evaluate_policy_promotion(
    baseline: dict,
    candidate: dict,
    *,
    min_exact_pp: float = 10.0,
) -> dict:
    """Apply the preregistered held-out gate before spending a GAIA Shadow exposure."""
    baseline_rate = float(baseline.get("exact_success_rate", 0.0))
    candidate_rate = float(candidate.get("exact_success_rate", 0.0))
    delta = round(candidate_rate - baseline_rate, 10)

    baseline_stops = baseline.get("stop_reasons") or {}
    candidate_stops = candidate.get("stop_reasons") or {}
    regressions = {
        reason: int(candidate_stops.get(reason, 0)) - int(baseline_stops.get(reason, 0))
        for reason in _CATASTROPHIC
        if int(candidate_stops.get(reason, 0)) > int(baseline_stops.get(reason, 0))
    }

    reasons: list[str] = []
    if delta < float(min_exact_pp):
        reasons.append(f"exact_success_gain_below_{float(min_exact_pp):.1f}pp")
    if regressions:
        reasons.append("catastrophic_failure_regression")

    return {
        "promote": not reasons,
        "baseline_exact_success_rate": baseline_rate,
        "candidate_exact_success_rate": candidate_rate,
        "exact_pp_delta": delta,
        "minimum_exact_pp": float(min_exact_pp),
        "catastrophic_regressions": regressions,
        "reasons": reasons,
    }


__all__ = ["evaluate_policy_promotion"]
