"""Prepare evidence-grounded 2WikiMultihopQA tasks without GAIA leakage.

The source dataset is the Apache-2.0 licensed 2WikiMultihopQA in Hotpot-like
schema: framolfese/2WikiMultihopQA, original Alab-NII/2wikimultihop.
This is an independent multi-hop QA source, not an externally scored GAIA run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from itertools import islice

from gaia_small_agent.training.protection import question_hash
from scripts.prepare_hotpotqa_tool_tasks import convert_row

DATASET = "framolfese/2WikiMultihopQA"


def convert_2wiki_row(row: dict, *, split: str) -> dict:
    if split not in {"train", "validation"}:
        raise ValueError("unsupported source split")
    # The 2Wiki authors release the same context/sentences and supporting-fact
    # layout as HotpotQA. The source/attribution MUST be overridden.
    task = convert_row(row, split=split)
    answer = task["expected_answer"].strip().casefold()
    if answer in {"yes", "no"}:
        raise ValueError("binary answer requires a separate entailment verifier")
    if not any(
        answer in str(content).casefold()
        for name, content in task["files"].items() if name.startswith("article_")
    ):
        raise ValueError("short answer not present in source articles")
    task.update({
        "id": f"2wiki-{split}-{row['id']}",
        "source": f"2wikimultihopqa-{split}",
        "license": "Apache-2.0",
        "generator": "2wiki-offline-read-agent",
        "generator_version": "1",
        "oracle_type": "human-annotated-exact-answer",
        "oracle_version": "2wikimultihopqa-coling2020-v1",
        "dataset_attribution": "Ho et al., 2WikiMultihopQA, COLING 2020; Alab-NII/2wikimultihop (Apache-2.0)",
    })
    return task


def select_2wiki_rows(raw_rows: list[dict], *, split: str, limit: int) -> tuple[list[dict], dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    chosen: list[dict] = []
    seen_ids: set[str] = set()
    rejected: Counter[str] = Counter()
    for raw in raw_rows:
        if len(chosen) >= limit:
            break
        try:
            row = convert_2wiki_row(raw, split=split)
            if row["id"] in seen_ids:
                raise ValueError("duplicate source task id")
            seen_ids.add(row["id"])
            chosen.append(row)
        except (KeyError, ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
    if not chosen:
        raise ValueError("no extractive multi-article tasks in scanned page")
    return chosen, {
        "source": DATASET, "split": split, "license": "Apache-2.0",
        "examined_until_acceptance": len(chosen) + sum(rejected.values()),
        "tasks": len(chosen), "rejected": dict(sorted(rejected.items())),
        "answer_hidden_from_agent_prompt": True,
        "article_text_untruncated": True,
    }


def fetch_rows(*, split: str, offset: int, scan_count: int) -> tuple[list[dict], str]:
    if split not in {"train", "validation"} or offset < 0 or scan_count < 1 or scan_count > 400:
        raise ValueError("invalid fetch bounds")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    # Resolve a concrete immutable revision before loading, record it in the
    # artifact. Network errors fail closed; never silently substitute a mirror.
    commit = HfApi().dataset_info(DATASET).sha
    if not commit or len(commit) != 40:
        raise ValueError("source dataset revision did not resolve")
    ds = load_dataset(DATASET, revision=commit, split=split, streaming=True)
    batch = list(islice(ds, offset, offset + scan_count))
    if not batch:
        raise ValueError("source yielded no data")
    return batch, commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--scan-count", type=int, default=250)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--holdout-hashes", type=Path)
    args = parser.parse_args()
    if (args.split == "train") == bool(args.holdout_hashes):
        raise ValueError("only validation split may produce holdout hashes, and it is required")
    outputs = [args.output, args.report] + ([args.holdout_hashes] if args.holdout_hashes else [])
    if any(path.exists() for path in outputs):
        raise FileExistsError("refuse to overwrite source or protected evidence")
    source, commit = fetch_rows(
        split=args.split, offset=args.offset, scan_count=args.scan_count)
    tasks, report = select_2wiki_rows(source, split=args.split, limit=args.limit)
    report.update({"dataset_commit_sha": commit,
                   "selection_hash": sha256(
                       "\n".join(task["id"] for task in tasks).encode()).hexdigest(),
                   "offset": args.offset, "scan_count": args.scan_count})
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if args.holdout_hashes:
        hashes = sorted({question_hash(task["question"]) for task in tasks})
        args.holdout_hashes.write_text(json.dumps({
            "algorithm": "sha256-normalized-question-v1", "hashes": hashes
        }, indent=2) + "\n")
    print("TWO_WIKI_PREP", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
