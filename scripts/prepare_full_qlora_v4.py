"""Prepare *all* clean training actions for a full 9B QLoRA run.

Unlike v3E (64-action diagnostic ablation), v4 uses every audited train-split
action, filters deliberately failed first searches globally, and keeps the
successfully recovered later tool/final states. DEV/TEST are NEVER included.
This is a full-compute synthetic policy SFT control, NOT a claim of real-world
generalization. Independent Hotpot/GAIA evaluation remains mandatory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def prepare_full_clean_train(rows: list[dict]) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("training source is empty")
    selected: list[dict] = []
    counts = Counter()
    task_ids: set[str] = set()
    for row in rows:
        if row.get("verified") is not True:
            raise ValueError("unverified training trajectory")
        if not str(row.get("source") or "").endswith("/train") or "gaia" in str(row.get("source", "")).lower():
            raise ValueError("only the independent oracle train split may be used; no dev/test/GAIA")
        verification = row.get("verification")
        if not isinstance(verification, dict) or any(
            verification.get(k) is not True
            for k in ("completed", "final_answer_correct", "zero_duplicate_blocks", "required_tools_satisfied", "provenance_complete")
        ):
            raise ValueError("incomplete process verification")
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in task_ids:
            raise ValueError("empty or duplicate task_id")
        task_ids.add(task_id)
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("missing messages")
        capability = str(row.get("capability") or "")
        allowed = []
        for index, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if capability in {"failure_recovery", "strategy_switch"} and index == 2 and msg.get("tool_calls"):
                # Deliberate SEARCH_FIXTURE_MISS. Never treat the known failed
                # first attempt as a correct next-action target.
                counts["removed_intentionally_bad_actions"] += 1
                continue
            allowed.append(index)
        if not allowed or messages[allowed[-1]].get("tool_calls"):
            raise ValueError(f"no valid final supervised action: {task_id}")
        clean = dict(row)
        clean["training_action_indices"] = allowed
        clean["curation"] = "global-bad-first-call-exclusion/v1"
        selected.append(clean)
        counts["total_trainable_actions"] += len(allowed)
        counts["tasks"] += 1
    report = {
        "schema": "full-synthetic-clean-qlora/v1",
        "source": "small-agent-policy-v3 TRAIN only",
        "tasks": len(selected),
        "total_trainable_actions": counts["total_trainable_actions"],
        "removed_intentionally_bad_actions": counts["removed_intentionally_bad_actions"],
        "capability_counts": dict(sorted(Counter(r["capability"] for r in selected).items())),
        "caveat": "Synthetic verified policy trajectories only. Improvement on GAIA or real tools is unknown."
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    cleaned, report = prepare_full_clean_train(rows)
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"refuse to replace existing training data: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in cleaned),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
