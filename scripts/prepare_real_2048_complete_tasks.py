"""Build a *whole-task*, untruncated 2048-token real multihop SFT dataset.

Do not silently crop evidence, drop only a long final answer, repeat tasks, use
GAIA labels, or infer GPU trainability from this CPU-only tokenizer preflight.
Actual GPU VRAM feasibility and independent task-success testing remain separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.training.qlora import load_verified_rows, _render_turn_examples


def filter_whole_tasks(
    rows: list[dict], examples: list[dict], tokenizer, max_length: int
) -> tuple[list[dict], dict]:
    if max_length <= 0 or not rows or not examples:
        raise ValueError("require nonempty tasks, examples and positive context budget")
    unique_ids = [str(row.get("task_id", "")) for row in rows]
    if not all(unique_ids) or len(set(unique_ids)) != len(unique_ids):
        raise ValueError("invalid or repeated parent task IDs")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        md = ex.get("metadata") or {}
        tid = str(md.get("task_id") or "")
        if tid not in unique_ids:
            raise ValueError("unknown rendered task identity")
        if not ex.get("completion"):
            raise ValueError("empty assistant completion")
        by_task[tid].append(ex)
    selected: list[dict] = []
    rejected = Counter()
    stats = Counter()
    selected_source = Counter()
    for row in rows:
        tid = str(row["task_id"])
        rendered = by_task.get(tid, [])
        declared = row.get("training_action_indices")
        if not isinstance(declared, list) or len(declared) != len(rendered):
            raise ValueError("incomplete native supervised actions for task " + tid)
        actual_indices = [e["metadata"]["assistant_message_index"] for e in rendered]
        if set(declared) != set(actual_indices) or len(set(actual_indices)) != len(actual_indices):
            raise ValueError("unmatched assistant action indices for task " + tid)
        if sum(e["metadata"]["action_type"] == "final" for e in rendered) != 1:
            raise ValueError("missing/duplicate final answer for task " + tid)
        lengths = []
        for ex in rendered:
            ids_prompt = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
            ids_completion = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
            if not ids_completion:
                raise ValueError("zero-token supervised completion")
            lengths.append(len(ids_prompt) + len(ids_completion))
        if max(lengths) > max_length:
            rejected["at_least_one_action_exceeds_context"] += 1
            stats["rejected_actions"] += len(lengths)
            continue  # Reject the ENTIRE parent, including short tool calls.
        if not str(row.get("source", "")).endswith("-train"):
            raise ValueError("non-training source")
        selected.append(row)
        selected_source[row["source"]] += 1
        stats["retained_actions"] += len(lengths)
        stats["retained_tool_calls"] += sum(
            ex["metadata"]["action_type"] == "tool_call" for ex in rendered
        )
        stats["retained_final_answers"] += 1
    report = {
        "schema": "multisource-whole-task-untruncated-v1",
        "context_budget_tokens": max_length,
        "input_unique_tasks": len(rows),
        "retained_unique_tasks": len(selected),
        "excluded_unique_tasks": len(rows) - len(selected),
        "retained_native_actions": stats["retained_actions"],
        "retained_tool_calls": stats["retained_tool_calls"],
        "retained_final_answers": stats["retained_final_answers"],
        "rejected_actions_with_parent": stats["rejected_actions"],
        "rejected_reason_counts": dict(sorted(rejected.items())),
        "source_task_counts": dict(sorted(selected_source.items())),
        "complete_parent_trajectories_only": True,
        "training_ran": False,
        "gaia_evaluation_ran": False,
        "limit": "CPU tokenizer fit is NOT GPU VRAM feasibility; read-only multihop tasks are narrow, and no accuracy gain is established.",
    }
    if stats["retained_actions"] != stats["retained_tool_calls"] + stats["retained_final_answers"]:
        raise AssertionError("native action supervision is not complete")
    return selected, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--protected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--expected-tasks", type=int, default=40)
    p.add_argument("--expected-actions", type=int, default=164)
    p.add_argument("--minimum-complete-tasks", type=int, default=39)
    a = p.parse_args()
    if a.output.resolve() == a.report.resolve() or a.output.exists() or a.report.exists():
        raise FileExistsError("never overwrite an observed training artifact")
    from transformers import AutoTokenizer
    rows = load_verified_rows(a.input, protected_questions_path=a.protected)
    if len(rows) != a.expected_tasks:
        raise ValueError("unexpected verified task count")
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    rendered = _render_turn_examples(tokenizer, rows)
    if len(rendered) != a.expected_actions:
        raise ValueError("unexpected native supervised-action count")
    selected, summary = filter_whole_tasks(rows, rendered, tokenizer, a.max_length)
    if len(selected) < a.minimum_complete_tasks:
        raise ValueError("too few entire full-context trajectories; do not pad or truncate")
    if summary["retained_final_answers"] != len(selected):
        raise AssertionError("lost verified final answer")
    if set(summary["source_task_counts"]) != {
        "hotpotqa-distractor-train", "2wikimultihopqa-train"
    }:
        raise ValueError("train source diversity gate failed")
    summary["model"] = a.model
    summary["input_sha256"] = sha256(a.input.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in selected),
        encoding="utf-8",
    )
    summary["output_sha256"] = sha256(a.output.read_bytes()).hexdigest()
    a.report.write_text(json.dumps(summary, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print("REAL_2048_COMPLETE_TASKS", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
