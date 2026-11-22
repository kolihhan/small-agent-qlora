"""Privacy-safe paired GAIA100 vs original v8 TRAIN trajectory depth study.

Observational only: step counts cannot establish the causal reason for
accuracy changes, or justify training on benchmark answers.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median


def _summarize(values):
    return {"n": len(values), "mean": round(sum(values) / len(values), 3),
            "median": median(values), "min": min(values), "max": max(values)}


def audit(train: Path, gaia_root: Path, *, strict=True):
    rows = [json.loads(line) for line in train.read_text("utf8").splitlines() if line.strip()]
    if strict and (len(rows) != 304 or sum(len(x["training_action_indices"]) for x in rows) != 798):
        raise ValueError("Original 304/798 TRAIN source drift")
    if not rows or any(x.get("verified") is not True for x in rows):
        raise ValueError("Unverified training source")
    train_depth = []
    final_actions = 0
    tool_actions = 0
    for row in rows:
        msgs = row["messages"]
        chosen = row["training_action_indices"]
        if len(chosen) != len(set(chosen)) or any(msgs[i]["role"] != "assistant" for i in chosen):
            raise ValueError("Unexpected training action indices")
        steps = sum(m["role"] == "assistant" for m in msgs)
        train_depth.append(steps)
        for idx in chosen:
            if msgs[idx].get("tool_calls"):
                tool_actions += 1
            else:
                final_actions += 1

    arms = {}
    for arm in ("base", "qlora-v8"):
        files = sorted(gaia_root.glob(f"gaia-v8-paired-shard-*/{arm}/shard-*/results.jsonl"))
        if strict and len(files) != 20:
            raise ValueError("Incomplete original 20 GAIA shards")
        seen = {}
        for file in files:
            for line in file.read_text("utf8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = row["task_id"]
                if tid in seen:
                    raise ValueError("Duplicate task ID in one arm")
                seen[tid] = {
                    "steps": int(row["steps"]), "tool_calls": int(row["tool_calls"]),
                    "correct": bool(row["correct"]), "stop_reason": str(row["stop_reason"])
                }
        if not seen:
            raise ValueError("No original model results")
        arms[arm] = seen
    if set(arms["base"]) != set(arms["qlora-v8"]):
        raise ValueError("GAIA arm mismatch")
    if strict and (len(arms["base"]) != 100
                   or sum(x["correct"] for x in arms["base"].values()) != 20
                   or sum(x["correct"] for x in arms["qlora-v8"].values()) != 21):
        raise ValueError("Historical matched GAIA20/21 identity drift")

    def arm_report(raw):
        values = list(raw.values())
        correct = [x for x in values if x["correct"]]
        wrong = [x for x in values if not x["correct"]]
        return {
            "n": len(values), "correct": len(correct), "steps": _summarize([x["steps"] for x in values]),
            "tool_calls": _summarize([x["tool_calls"] for x in values]),
            "steps_ge_8": sum(x["steps"] >= 8 for x in values),
            "steps_eq_12": sum(x["steps"] == 12 for x in values),
            "correct_tool_calls": _summarize([x["tool_calls"] for x in correct]) if correct else None,
            "incorrect_or_incomplete_tool_calls": _summarize([x["tool_calls"] for x in wrong]) if wrong else None,
            "stops": dict(Counter(x["stop_reason"] for x in values)),
        }

    paired = Counter()
    delta = {}
    for tid, a in arms["base"].items():
        b = arms["qlora-v8"][tid]
        status = ("rescue" if not a["correct"] and b["correct"] else
                  "regression" if a["correct"] and not b["correct"] else
                  "both_correct" if a["correct"] else "both_wrong")
        paired[status] += 1
        delta.setdefault(status, []).append(b["steps"] - a["steps"])
    return {
        "status": "HISTORICAL_CPU_DEPTH_COVERAGE_ONLY_NOT_TRAINING",
        "train": {"n": len(rows), "steps": _summarize(train_depth),
                  "steps_ge_8": sum(x >= 8 for x in train_depth),
                  "tool_supervision": tool_actions, "final_supervision": final_actions},
        "gaia": {arm: arm_report(records) for arm, records in arms.items()},
        "paired_outcomes": dict(sorted(paired.items())),
        "paired_step_delta": {name: _summarize(numbers) for name, numbers in sorted(delta.items())},
        "real_model_calls": 0, "training_ran": False, "new_gaia_score": False,
        "no_task_answers_or_prompts_exported": True,
        "warning": "Depth mismatch is not a causal training deficiency; GAIA100 is historically exposed.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--gaia-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = audit(args.train, args.gaia_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("V8_TRAIN_GAIA_DEPTH_AUDIT", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
