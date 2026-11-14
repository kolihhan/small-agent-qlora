"""Prepare independent FinQA TRAIN numerical tool-use candidates for QLoRA.

Only source-documented real financial tasks. Excludes *all* FinQA DEV report IDs
and question text, plus percent/ambiguous labels. No pseudo teacher actions:
eligible source tasks still require actual clean observed executions before SFT.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from scripts.finqa_numeric_v7_transfer20 import QUESTION_CUE, close, number, report_text

SOURCE_REV = "0f16e2867befa6840783e58be38c9efb9229d742"
PILOT_SEED = "finqa-observed-multitool-pilot-20261114"


def select_training_tasks(train: list[dict], dev: list[dict], *,
                          limit: int = 16) -> list[dict]:
    if not 1 <= limit <= 64:
        raise ValueError("pilot count outside bounded range")
    heldout_docs = {str(row.get("filename")) for row in dev}
    heldout_questions = {" ".join(str(r.get("qa", {}).get("question") or "").casefold().split())
                         for r in dev}
    candidates = []
    seen = set()
    for row in train:
        qa = row.get("qa") or {}
        report_id = str(row.get("filename") or "")
        question = str(qa.get("question") or "")
        qhash = " ".join(question.casefold().split())
        gold = number(str(qa.get("answer") or ""))
        exe = qa.get("exe_ans")
        if not report_id or report_id in heldout_docs or not 25 <= len(question) <= 300:
            continue
        if qhash in heldout_questions or qhash in seen or not QUESTION_CUE.search(question):
            continue
        if (gold is None or gold[1] or not isinstance(exe, (int, float))
            or isinstance(exe, bool) or not close(gold[0], float(exe))
            or len(qa.get("steps") or []) < 2):
            continue
        try:
            evidence = report_text(row)
        except (ValueError, TypeError):
            continue
        if not 100 <= len(evidence) <= 12000:
            continue
        seen.add(qhash)
        candidates.append((row, evidence))
    ordered = sorted(candidates, key=lambda item: (
        sha256(f"{PILOT_SEED}:{item[0]['id']}".encode()).hexdigest(),
        str(item[0]["id"]),
    ))
    out, used_reports = [], set()
    for row, evidence in ordered:
        report_id = str(row["filename"])
        if report_id in used_reports:
            continue
        used_reports.add(report_id)
        out.append({
            "id": f"finqa-train-{row['id']}",
            "source": "finqa-train",
            "license": "MIT",
            "generator": "official-finqa-report-multitool-task",
            "generator_version": SOURCE_REV,
            "oracle_type": "human-annotated-and-program-numeric-consistent",
            "oracle_version": "FinQA-EMNLP2021",
            "question": (
                "Answer the financial calculation using ONLY the local file report.txt. "
                "Read report.txt, use python for arithmetic, and return one short numeric "
                "answer without explanation. Question: " + row["qa"]["question"]
            ),
            "expected_answer": str(row["qa"]["answer"]).strip(),
            "required_tools": ["read", "python"],
            "files": {"report.txt": evidence},
            "document_id_hash": sha256(report_id.encode()).hexdigest(),
            "program_steps": len(row["qa"]["steps"]),
        })
        if len(out) == limit:
            break
    if len(out) != limit or len({r["document_id_hash"] for r in out}) != limit:
        raise ValueError(f"insufficient report-disjoint FinQA TRAIN tool tasks {len(out)}/{limit}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--dev", required=True, type=Path)
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or args.report.exists() or args.output.resolve() == args.report.resolve():
        raise FileExistsError("refuse overwrite")
    train_bytes, dev_bytes = args.train.read_bytes(), args.dev.read_bytes()
    rows = select_training_tasks(json.loads(train_bytes), json.loads(dev_bytes),
                                 limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, sort_keys=True, ensure_ascii=False)+"\n"
                                   for x in rows), encoding="utf-8")
    report = {
        "selected": len(rows), "distinct_documents": len({r["document_id_hash"] for r in rows}),
        "source_revision": SOURCE_REV, "source_train_sha": sha256(train_bytes).hexdigest(),
        "source_dev_sha": sha256(dev_bytes).hexdigest(), "gaia_examples": 0,
        "verified_observed_trajectories": 0, "training_ran": False,
        "source_family": "finqa-train", "required_tools": ["read", "python"],
        "mean_program_steps": sum(r["program_steps"] for r in rows)/len(rows),
        "boundary": "Candidate tasks only; must execute and independently verify before QLoRA training",
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("FINQA_TEACHER_SOURCE_READY", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
