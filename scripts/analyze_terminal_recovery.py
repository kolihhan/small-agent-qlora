"""Paired, aggregate-only GAIA termination diagnosis.

Run locally on matched Base/v3B/v3D results.jsonl. Do not upload or commit
protected GAIA rows, raw traces, task IDs, questions, or model answers.

This script only analyzes observed behavior; it is not a counterfactual
evaluation or evidence that a new model is better.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import chain
import json
from math import comb
from pathlib import Path


def _load(path: Path, expected: int) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("result must be a JSON object")
        ident = row.get("task_id")
        if not isinstance(ident, str) or not ident:
            raise ValueError("missing task identity")
        if ident in rows:
            raise ValueError("duplicate task identity")
        if not isinstance(row.get("completed"), bool) or not isinstance(row.get("correct"), bool):
            raise ValueError("completed/correct must be boolean")
        if row["correct"] and not row["completed"]:
            raise ValueError("an incomplete answer cannot be correct")
        if not isinstance(row.get("steps"), int) or isinstance(row["steps"], bool) or not 1 <= row["steps"] <= 12:
            raise ValueError("steps must be an integer between 1 and 12")
        for key in ("tool_calls", "duplicate_calls_blocked", "tool_errors"):
            if not isinstance(row.get(key), int) or isinstance(row[key], bool) or row[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not isinstance(row.get("stop_reason"), str):
            raise ValueError("stop_reason is required")
        if not isinstance(row.get("trace"), list):
            raise ValueError("trace must be a list")
        rows[ident] = row
    if len(rows) != expected:
        raise ValueError(f"expected {expected} unique rows, got {len(rows)}")
    return rows


def _bin(steps: int) -> str:
    if steps <= 3:
        return "01-03"
    if steps <= 6:
        return "04-06"
    if steps <= 9:
        return "07-09"
    return "10-12"


def _tool_markup_final(row: dict) -> bool:
    # Only a proxy for the final-output parser failure; never return raw content.
    if row["stop_reason"] != "invalid_final":
        return False
    for event in reversed(row["trace"]):
        if isinstance(event, dict) and event.get("kind") == "invalid_final":
            data = event.get("data")
            return isinstance(data, dict) and str(data.get("content", "")).lstrip().startswith("<tool_call>")
    return False


def _mcnemar_exact(b_only: int, d_only: int) -> float:
    n = b_only + d_only
    if n == 0:
        return 1.0
    low = min(b_only, d_only)
    return min(1.0, 2.0 * sum(comb(n, k) for k in range(low + 1)) / (2 ** n))


def _arm(rows: list[dict]) -> dict:
    n = len(rows)
    steps = Counter(_bin(row["steps"]) for row in rows)
    reasons = Counter(row["stop_reason"] for row in rows)
    return {
        "n": n,
        "correct": sum(row["correct"] for row in rows),
        "completed": sum(row["completed"] for row in rows),
        "completion_rate": sum(row["completed"] for row in rows) / n if n else None,
        "tool_calls": sum(row["tool_calls"] for row in rows),
        "duplicate_calls_blocked": sum(row["duplicate_calls_blocked"] for row in rows),
        "tool_errors": sum(row["tool_errors"] for row in rows),
        "steps_by_bucket": dict(sorted(steps.items())),
        "stop_reasons": dict(sorted(reasons.items())),
        "step12_invalid_final": sum(row["steps"] == 12 and row["stop_reason"] == "invalid_final" for row in rows),
        "textual_tool_markup_invalid_final": sum(_tool_markup_final(row) for row in rows),
    }


def _paired_counts(pairs: list[tuple[dict, dict]], field: str) -> dict:
    both_yes = sum(b[field] and d[field] for b, d in pairs)
    b_only = sum(b[field] and not d[field] for b, d in pairs)
    d_only = sum(not b[field] and d[field] for b, d in pairs)
    both_no = len(pairs) - both_yes - b_only - d_only
    return {
        "both_yes": both_yes,
        "baseline_only": b_only,
        "recovery_only": d_only,
        "both_no": both_no,
        "exact_mcnemar_p": _mcnemar_exact(b_only, d_only),
    }


def summarize(baseline: dict[str, dict], recovery: dict[str, dict]) -> dict:
    if set(baseline) != set(recovery):
        raise ValueError("paired task identities differ")
    # Paired groupings are based ONLY on baseline v3B attributes.
    pairs = [(baseline[k], recovery[k]) for k in sorted(baseline)]
    cohorts: dict[str, dict] = {}
    for bucket in ("01-03", "04-06", "07-09", "10-12"):
        subset = [(b, d) for b, d in pairs if _bin(b["steps"]) == bucket]
        cohorts[bucket] = {
            "n": len(subset),
            "baseline_completed": sum(b["completed"] for b, _ in subset),
            "recovery_completed": sum(d["completed"] for _, d in subset),
            "baseline_correct": sum(b["correct"] for b, _ in subset),
            "recovery_correct": sum(d["correct"] for _, d in subset),
            "finalization_transitions": _paired_counts(subset, "completed"),
        }
    return {
        "schema": "qlora-terminal-recovery-v1",
        "n": len(pairs),
        "baseline": _arm([b for b, _ in pairs]),
        "recovery": _arm([d for _, d in pairs]),
        "paired_completion": _paired_counts(pairs, "completed"),
        "paired_correctness": _paired_counts(pairs, "correct"),
        "cohorts_by_baseline_steps": cohorts,
        "interpretation_warning": (
            "Descriptive paired analysis only. Cohorting by observed steps is not causal; "
            "conditioning on completion selects a biased subset. No task-level or gated text is emitted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.expected <= 1000:
        parser.error("--expected must be 1..1000")
    result = summarize(_load(args.baseline, args.expected), _load(args.recovery, args.expected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Aggregated {result['n']} paired tasks into {args.output.name}; no task-level output.")


if __name__ == "__main__":
    main()
