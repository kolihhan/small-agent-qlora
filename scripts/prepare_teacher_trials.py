"""Prepare independent best-of-N non-GAIA tasks for the existing tool collector.

Example:
  python -m scripts.prepare_teacher_trials --tasks tasks/train.jsonl --samples 3 --output runs/teacher-trials.jsonl
  small-agent collect-trajectories --tasks runs/teacher-trials.jsonl --output runs/teacher-observed.jsonl \\
    --protected-questions runs/gaia-protected-question-hashes.json [backend flags]
  python -m gaia_small_agent.training.teacher_curation --input runs/teacher-observed.jsonl \\
    --protected runs/gaia-protected-question-hashes.json --holdout-hashes runs/holdout-question-hashes.json \\
    --output runs/teacher-curated.jsonl --report runs/teacher-curated-summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_TAG = "::sample-"


def prepare_trials(tasks: list[dict], *, samples: int) -> list[dict]:
    if not 2 <= samples <= 8:
        raise ValueError("samples must be in range 2..8")
    ids: set[str] = set()
    expanded: list[dict] = []
    for task in tasks:
        task_id = task.get("id")
        source = str(task.get("source") or "")
        if not isinstance(task_id, str) or not task_id or _TAG in task_id:
            raise ValueError("original task id must be nonempty and not contain sample delimiter")
        if task_id in ids:
            raise ValueError("duplicate original task id")
        if not source or "gaia" in source.casefold():
            raise ValueError("GAIA and unknown-source tasks cannot be trial-expanded")
        for key in ("question", "expected_answer"):
            if not isinstance(task.get(key), str) or not task[key]:
                raise ValueError(f"task missing {key}")
        for key in ("license", "generator", "generator_version", "oracle_type", "oracle_version"):
            if not task.get(key):
                raise ValueError(f"missing task provenance {key}")
        ids.add(task_id)
        for trial in range(1, samples + 1):
            copied = dict(task)
            copied["id"] = f"{task_id}{_TAG}{trial}"
            expanded.append(copied)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    expanded = prepare_trials(tasks, samples=args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite existing teacher trials")
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in expanded),
        encoding="utf-8",
    )
    print(json.dumps({"original_tasks": len(tasks), "trials": len(expanded), "samples_per_task": args.samples}))


if __name__ == "__main__":
    main()
