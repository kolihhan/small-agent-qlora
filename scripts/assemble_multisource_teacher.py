"""Merge real observed multi-hop agent trials with one strict cross-source curator.

Never uses GAIA reference answers, validation questions or fabricated trajectories.
The mixed source-gate is an *eligibility* check, not a claim of generalization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.training.protection import load_protected_question_hashes
from gaia_small_agent.training.teacher_curation import curate_teacher_rows

_ALLOWED_SOURCES = {"hotpotqa-distractor-train", "2wikimultihopqa-train"}


def _load_observed(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    if not paths or len(paths) != len({str(p.resolve()) for p in paths}):
        raise ValueError("input paths empty or duplicated")
    all_rows: list[dict] = []
    manifest: list[dict] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"no observed verified rows: {path.name}")
        for row in rows:
            if not isinstance(row, dict) or row.get("verified") is not True:
                raise ValueError(f"nonverified row in {path.name}")
            if row.get("source") not in _ALLOWED_SOURCES:
                raise ValueError(f"unknown or forbidden source: {row.get('source')}")
            task_id = str(row.get("task_id") or "")
            if not task_id or task_id in seen_ids:
                raise ValueError(f"duplicate/missing task identity {task_id}")
            seen_ids.add(task_id)
        all_rows.extend(rows)
        manifest.append({
            "file_name": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "verified_rows": len(rows),
            "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        })
    return all_rows, manifest


def assemble(*, inputs: list[Path], protected: Path, holdouts: list[Path],
             min_unique_tasks: int = 24,
             min_tasks_per_source: int = 1) -> tuple[list[dict], dict]:
    if min_unique_tasks < 1 or min_tasks_per_source < 1:
        raise ValueError("positive evidence gates required")
    protected_hashes = load_protected_question_hashes(protected)
    holdout_hashes: set[str] = set()
    if len(holdouts) != 2 or len({str(p.resolve()) for p in holdouts}) != 2:
        raise ValueError("explicit HotpotQA and 2Wiki validation exclusion sets required")
    for holdout in holdouts:
        hashes = load_protected_question_hashes(holdout)
        if not hashes:
            raise ValueError(f"empty independent holdout hash set: {holdout.name}")
        holdout_hashes.update(hashes)
    if len(protected_hashes) < 100:
        raise ValueError("real GAIA protection set required, not a test-only placeholder")
    rows, input_manifest = _load_observed(inputs)
    selected, curation = curate_teacher_rows(
        rows, protected_hashes=protected_hashes,
        holdout_hashes=holdout_hashes, min_actions=3, min_long_trajectories=1,
    )
    task_counts = Counter(row["source"] for row in selected)
    if len(selected) < min_unique_tasks:
        raise ValueError(f"insufficient independently curated tasks {len(selected)} < {min_unique_tasks}")
    if any(task_counts[s] < min_tasks_per_source for s in sorted(_ALLOWED_SOURCES)):
        raise ValueError(f"insufficient cross-source evidence: {dict(task_counts)}")
    if any(row["selection_provenance"]["tool_calls"] < 3 for row in selected):
        raise AssertionError("curator accepted too-shallow tool trajectory")
    if len({row["task_id"] for row in selected}) != len(selected):
        raise AssertionError("curator duplicated independent parent IDs")
    report = {
        "schema": "multi-source-strict-real-teacher-v1",
        "input_verified_rollouts": len(rows),
        "accepted_unique_tasks": len(selected),
        "accepted_task_source_counts": dict(sorted(task_counts.items())),
        "trainable_assistant_actions": sum(len(row["training_action_indices"]) for row in selected),
        "source_manifest": input_manifest,
        "strict_curator": curation,
        "min_unique_tasks_gate": min_unique_tasks,
        "min_tasks_per_source_gate": min_tasks_per_source,
        "training_ran": False,
        "gaia_evaluation_ran": False,
        "caveat": "Extractive observed-answer and structural tool checks are not entailment, task coverage, or GAIA generalization guarantees.",
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--holdout-hashes", type=Path, nargs=2, required=True)
    parser.add_argument("--min-unique-tasks", type=int, default=24)
    parser.add_argument("--min-tasks-per-source", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve() or args.output.exists() or args.report.exists():
        raise FileExistsError("refuse overwriting curated evidence")
    selected, report = assemble(
        inputs=args.inputs, protected=args.protected,
        holdouts=args.holdout_hashes, min_unique_tasks=args.min_unique_tasks,
        min_tasks_per_source=args.min_tasks_per_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in selected),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("MULTISOURCE_STRICT_FINAL", json.dumps({
        k:v for k,v in report.items() if k not in {"source_manifest", "strict_curator"}},
        sort_keys=True, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
