"""Merge aggregate-only matched Shadow25 shard diagnostics without exposing tasks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.analyze_terminal_recovery import _mcnemar_exact

ARM_COUNT_KEYS = (
    "n", "correct", "completed", "tool_calls", "duplicate_calls_blocked",
    "tool_errors", "step12_invalid_final", "textual_tool_markup_invalid_final",
)
PAIR_COUNT_KEYS = ("both_yes", "baseline_only", "recovery_only", "both_no")
COHORT_COUNT_KEYS = ("n", "baseline_completed", "recovery_completed", "baseline_correct", "recovery_correct")
BUCKETS = ("01-03", "04-06", "07-09", "10-12")


def _pair_sum(parts: list[dict]) -> dict:
    result = {key: sum(int(p[key]) for p in parts) for key in PAIR_COUNT_KEYS}
    result["exact_mcnemar_p"] = _mcnemar_exact(result["baseline_only"], result["recovery_only"])
    return result


def merge(shards: list[dict], *, expected_shards: int = 5, expected_total: int = 25) -> dict:
    if len(shards) != expected_shards:
        raise ValueError("unexpected shard count")
    if any(s.get("schema") != "qlora-terminal-recovery-v1" for s in shards):
        raise ValueError("unexpected shard schema")
    totals = [s["n"] for s in shards]
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in totals):
        raise ValueError("invalid shard population")
    if sum(totals) != expected_total:
        raise ValueError("unexpected aggregate population")
    output = {
        "schema": "qlora-shadow25-recovery/v1",
        "n": expected_total,
        "shards": expected_shards,
        "provenance": {
            "partition": "shadow",
            "experiment": "paired v3B vs v3D / Qwen3.5-9B / 12 agent steps",
            "evaluation_status": "local held-out diagnostic, not official GAIA",
        },
    }
    for arm in ("baseline", "recovery"):
        parts = [s[arm] for s in shards]
        a = {key: sum(int(p[key]) for p in parts) for key in ARM_COUNT_KEYS}
        if a["n"] != expected_total:
            raise ValueError("arm count mismatch")
        a["completion_rate"] = a["completed"] / expected_total
        for key in ("steps_by_bucket", "stop_reasons"):
            counts = Counter()
            for part in parts:
                counts.update(part[key])
            a[key] = dict(sorted(counts.items()))
        if sum(a["steps_by_bucket"].values()) != expected_total:
            raise ValueError("step bucket coverage mismatch")
        output[arm] = a
    for name in ("paired_completion", "paired_correctness"):
        output[name] = _pair_sum([s[name] for s in shards])
        if sum(output[name][k] for k in PAIR_COUNT_KEYS) != expected_total:
            raise ValueError("paired count mismatch")
    cohorts = {}
    for bucket in BUCKETS:
        parts = [s["cohorts_by_baseline_steps"][bucket] for s in shards]
        c = {key: sum(int(p[key]) for p in parts) for key in COHORT_COUNT_KEYS}
        c["finalization_transitions"] = _pair_sum([p["finalization_transitions"] for p in parts])
        if sum(c["finalization_transitions"][k] for k in PAIR_COUNT_KEYS) != c["n"]:
            raise ValueError("cohort paired count mismatch")
        cohorts[bucket] = c
    if sum(c["n"] for c in cohorts.values()) != expected_total:
        raise ValueError("cohort population mismatch")
    output["cohorts_by_baseline_steps"] = cohorts
    output["interpretation_warning"] = (
        "Observational paired Shadow25 test. Do not tune on Shadow25 and call it independent again. "
        "No raw gated GAIA questions, answers, task IDs, or traces are included."
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge([json.loads(p.read_text(encoding="utf-8")) for p in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Sanitized matched Shadow25 aggregate: {result['n']} tasks, {result['shards']} shards.")


if __name__ == "__main__":
    main()
