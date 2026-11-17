"""Audit original *verified* v8 QLoRA selected next-action horizon coverage.

Reads source-only TRAIN and source-disjoint teacher-forced heldout rows, never
GAIA validation content or adapter metrics. No inference, model, GPU or training.
The audit tests which prefix decision states the current supervision actually
covers, not whether those labels teach useful policies or improve GAIA.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

TRAIN_SHA = "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
HOLDOUT_SHA = "cf4560d3a88059aa0d0fb00467d86b36d696dcc6e6c3cc4208637c13bcdbe8b7"
PROTOCOL = "SOURCE-ONLY-V8-TRAIN-HOLDOUT-NEXT-ACTION-HORIZON-AUDIT-V1"
EXPECTED = {
    "train": {"parents": 304, "actions": 798, "late_tool": 11, "late_final": 68,
              "max_prefix_tool_turns": 5},
    "holdout": {"parents": 35, "actions": 176, "late_tool": 12, "late_final": 24,
                "max_prefix_tool_turns": 5},
}


def load_verified(path: Path, pinned_sha: str) -> list[dict]:
    data = path.read_bytes()
    if sha256(data).hexdigest() != pinned_sha:
        raise ValueError("original pinned QLoRA source SHA mismatch")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len({str(row.get("task_id") or "") for row in rows}) != len(rows):
        raise ValueError("duplicate training/holdout parent identity")
    for row in rows:
        if row.get("verified") is not True:
            raise ValueError("source contains unverified training labels")
        if "gaia" in str(row.get("source") or "").casefold():
            raise ValueError("GAIA-derived material cannot be counted as training source")
    return rows


def selected_action_states(row: dict) -> list[tuple[str, int]]:
    messages = row.get("messages")
    indices = row.get("training_action_indices")
    if (not isinstance(messages, list) or not isinstance(indices, list)
        or not indices or indices != sorted(set(indices))):
        raise ValueError("missing or malformed supervised action indices")
    states: list[tuple[str, int]] = []
    for idx in indices:
        if not isinstance(idx, int) or idx <= 0 or idx >= len(messages):
            raise ValueError("selected action index out of bounds")
        message = messages[idx]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("selected label not assistant next-action")
        # Count previous tool-action *turns*, not individual calls. Existing
        # train corpus has three assistant turns with two tool calls.
        prefix = sum(
            bool(x.get("tool_calls"))
            for x in messages[:idx]
            if isinstance(x, dict) and x.get("role") == "assistant"
        )
        kind = "tool" if message.get("tool_calls") else "final"
        states.append((kind, prefix))
    return states


def summarize(rows: list[dict]) -> dict:
    by_source: dict[str, Counter] = defaultdict(Counter)
    histogram: Counter = Counter()
    tasks_by_source: Counter = Counter()
    for row in rows:
        source = str(row.get("source") or "")
        tasks_by_source[source] += 1
        for kind, depth in selected_action_states(row):
            by_source[source][(kind, depth)] += 1
            histogram[(kind, depth)] += 1
    tool = sum(value for (kind, _), value in histogram.items() if kind == "tool")
    final = sum(value for (kind, _), value in histogram.items() if kind == "final")
    lt = sum(value for (kind, d), value in histogram.items() if kind == "tool" and d >= 4)
    lf = sum(value for (kind, d), value in histogram.items() if kind == "final" and d >= 4)
    return {
        "parents": len(rows),
        "actions": tool + final,
        "tool_labels": tool,
        "final_labels": final,
        "late_tool": lt,
        "late_final": lf,
        "late_total": lt + lf,
        "late_continue_fraction": round(lt / (lt + lf), 6) if (lt + lf) else None,
        "max_prefix_tool_turns": max((d for _, d in histogram), default=None),
        "selected_actions_at_prefix_6_or_more": sum(v for (_, d), v in histogram.items() if d >= 6),
        "histogram": {
            f"{kind}@{depth}": count
            for (kind, depth), count in sorted(histogram.items())
        },
        "sources": {
            source: {
                "parents": tasks_by_source[source],
                "labels": sum(by_source[source].values()),
                "late_tool": sum(v for (kind, d), v in by_source[source].items() if kind == "tool" and d >= 4),
                "late_final": sum(v for (kind, d), v in by_source[source].items() if kind == "final" and d >= 4),
            }
            for source in sorted(tasks_by_source)
        },
    }


def audit(train: Path, holdout: Path) -> dict:
    t = load_verified(train, TRAIN_SHA)
    h = load_verified(holdout, HOLDOUT_SHA)
    t_ids = {str(r["task_id"]) for r in t}
    h_ids = {str(r["task_id"]) for r in h}
    if t_ids.intersection(h_ids):
        raise ValueError("training and heldout identities overlap")
    a, b = summarize(t), summarize(h)
    for name, actual in (("train", a), ("holdout", b)):
        for key, expected in EXPECTED[name].items():
            if actual[key] != expected:
                raise ValueError(f"pinned {name} selected decision distribution drift: {key}")
    return {
        "status": "VERIFIED_CPU_ONLY_SOURCE_HORIZON_DIAGNOSTIC",
        "protocol": PROTOCOL,
        "input_shas": {"train": TRAIN_SHA, "holdout": HOLDOUT_SHA},
        "no_gaia_training": True,
        "no_model_inference": True,
        "no_training": True,
        "train": a,
        "holdout": b,
        "late_continuation_fraction_gap": round(b["late_continue_fraction"] - a["late_continue_fraction"], 6),
        "causal_training_gain_established": False,
        "proposed_gate": (
            "Before any new GPU/QLoRA run, require source-verified correct long-horizon "
            "next-action tool decisions at prefix>=4, including prefix>=6. Labeling "
            "unsubstantiated oracle reads or recycling exposed GAIA questions is prohibited. "
            "Pre-freeze task-family-diverse autonomy test before evaluating any adapter."
        ),
        "limitations": [
            "Selection histograms count supervised assistant turns, not independent tasks.",
            "Teacher-forced holdout is NOT an autonomous agent benchmark.",
            "Late continuation labels being scarce is a coverage observation, not causal proof.",
            "Identical source-family names do not establish independent verified reasoning labels."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite original evidence")
    result = audit(args.train, args.holdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("V8_LONG_HORIZON_COVERAGE_VERIFIED", json.dumps({
        key: result[key] for key in ("status", "input_shas", "late_continuation_fraction_gap")
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
