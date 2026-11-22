"""Sufficiency-proxy falsification on frozen verified v8 TRAIN.

No GAIA answers. Exact answer appearing in an observation is NOT sufficient
to decide stop, especially on multihop tasks.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).casefold()).strip()


def summarize(train: Path, *, expected_n: int = 304) -> dict:
    rows = [json.loads(line) for line in train.read_text("utf-8").splitlines() if line.strip()]
    if len(rows) != expected_n or any(r.get("verified") is not True for r in rows):
        raise ValueError("Frozen verified v8 train rows missing")
    aggregate: dict[str, Counter] = defaultdict(Counter)
    seen_ids: set[str] = set()
    total_actions = 0
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen_ids:
            raise ValueError("Missing or duplicate TRAIN task id")
        seen_ids.add(task_id)
        messages = row["messages"]
        selected = row["training_action_indices"]
        if len(set(selected)) != len(selected):
            raise ValueError("Repeated training action")
        if any(messages[i]["role"] != "assistant" for i in selected):
            raise ValueError("Training action index not an assistant turn")
        total_actions += len(selected)
        action_indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        if not action_indices or messages[action_indices[-1]].get("tool_calls"):
            raise ValueError("Missing terminal final")
        answer = _norm(messages[action_indices[-1]].get("content", ""))
        if not answer:
            raise ValueError("Empty final in verified trace")
        source = row["source"]
        counts = aggregate[source]
        counts["parents"] += 1
        counts["actions"] += len(action_indices)
        counts["selected_actions"] += len(selected)
        seen_answer = False
        any_premature = False
        after_answer_tool_steps = 0
        prior_evidence = ""
        for n, i in enumerate(action_indices):
            action = messages[i]
            answer_visible = bool(answer in _norm(prior_evidence))
            if answer_visible and n == len(action_indices)-1:
                counts["answer_verbatim_at_final_prefix"] += 1
            if answer_visible and action.get("tool_calls"):
                after_answer_tool_steps += 1
                any_premature = True
            nxt = action_indices[n + 1] if n + 1 < len(action_indices) else len(messages)
            prior_evidence += " " + " ".join(str(m.get("content", "")) for m in messages[i+1:nxt] if m["role"]=="tool")
        if not any(messages[i]["role"]=="tool" for i in range(len(messages))):
            counts["parents_with_no_tools"] += 1
        if any_premature:
            counts["answer_already_visible_while_agent_continued"] += 1
            counts["extra_tool_actions_after_answer_visible"] += after_answer_tool_steps
    if expected_n == 304 and total_actions != 798:
        raise ValueError("Original 798 training action drift")
    by_source = {k: dict(sorted(v.items())) for k, v in sorted(aggregate.items())}
    all_counts = sum(aggregate.values(), Counter())
    report = {
        "status": "HISTORICAL_TRAIN304_STOP_PROXY_FALSIFICATION_ONLY",
        "sources": by_source,
        "overall": dict(sorted(all_counts.items())),
        "negative_control": {
            "reason": "A verified training task can need additional tool actions even after the correct final answer string first appears in an observation.",
            "observed_counterexamples": all_counts["answer_already_visible_while_agent_continued"],
            "additional_tool_actions": all_counts["extra_tool_actions_after_answer_visible"],
        },
        "no_gaia_labels": True, "no_reference_answers_exported": True,
        "no_qwen_inference": True, "no_training": True,
        "promotion_allowed": False,
        "limitations": [
            "Contains-answer is a lexical proxy, not an evidence sufficiency label.",
            "Continuing after seeing the answer may be beneficial or redundant; there is no counterfactual success label.",
            "MuSiQue action indices are annotation-selected, not observed model behavior.",
            "A valid stop/continue classifier needs independent completed task outcomes and a clean evaluation set."
        ]
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = summarize(args.train)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("STOP_PROXY_FALSIFICATION", json.dumps({
        "status": report["status"], "overall": report["overall"],
        "sources": report["sources"]}, sort_keys=True))


if __name__ == "__main__":
    main()
