"""Post-hoc MuSiQue independent OTHER8 read-path coverage against pinned source support labels.

Runs AFTER Qwen decisions are frozen. Does not change prompts, labels or models.
Reading a supporting FILE is only an upper bound on delivered evidence: 2000-char
observations may truncate the relevant answer span.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.eval_qwen9b_musique_other8_readpath import (
    freeze_other, EXPECTED_MODEL_DIGEST, PROTOCOL as MODEL_PROTOCOL,
)
from scripts.prepare_v8_closedloop_holdout import (
    _full_files, SOURCE_LFS_SHA256, SOURCE_REV,
)

PROTOCOL = "qwen9b-musique-other8-annotated-support-path-audit-v1"
READ_PATH = re.compile(r"^passages/p([0-9]+)\.txt$")


def supporting_path_stats(source: dict, observed: dict) -> dict:
    paragraphs = source.get("paragraphs")
    if not isinstance(paragraphs, list) or len(paragraphs) != 20:
        raise ValueError("MuSiQue source requires 20 original paragraphs")
    ids = []
    for p in paragraphs:
        if type(p.get("idx")) is not int or type(p.get("is_supporting")) is not bool:
            raise ValueError("source annotation support labels missing/invalid")
        if p["is_supporting"]:
            ids.append(p["idx"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("no valid distinct annotated supporting paragraphs")
    read_paths = observed.get("read_paths")
    if not isinstance(read_paths, list):
        raise ValueError("actual model read-path log absent")
    visited = set()
    for path in read_paths:
        if not isinstance(path,str):
            raise ValueError("read path must be a string")
        match = READ_PATH.fullmatch(path)
        if match:
            ix = int(match.group(1))
            if ix not in {x["idx"] for x in paragraphs}:
                raise ValueError("read refers to nonexistent paragraph")
            visited.add(ix)
    relevant = set(ids)
    matched = sorted(relevant & visited)
    return {
        "annotated_support_passages": sorted(relevant),
        "actually_read_paragraph_paths": sorted(visited),
        "support_paths_visited": matched,
        "support_path_recall_upper_bound": round(len(matched)/len(relevant),4),
        "all_support_paths_visited": relevant <= visited,
        "missing_annotated_paths": sorted(relevant - visited),
        "distinct_paths_read": len(visited),
        "correct": bool(observed.get("correct")),
        "completed": bool(observed.get("completed")),
    }


def audit(original_source: Path, frozen_tasks: Path, frozen_report: Path,
          model_result: Path) -> dict:
    if sha256(original_source.read_bytes()).hexdigest() != SOURCE_LFS_SHA256:
        raise ValueError("original complete MuSiQue physical source checksum drift")
    task_rows = [json.loads(line) for line in frozen_tasks.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    report = json.loads(frozen_report.read_text(encoding="utf-8"))
    expected = freeze_other(task_rows, report)
    by_task = {r["task_id"]:r for r in expected}
    model = json.loads(model_result.read_text(encoding="utf-8"))
    if (model.get("protocol") != MODEL_PROTOCOL or model.get("total") != 8
        or len(model.get("cases",[])) != 8 or model.get("model_improvement_claim") is not False):
        raise ValueError("not the frozen OTHER8 independent 8-task model run")
    if set(by_task) != {r.get("task_id") for r in model["cases"]}:
        raise ValueError("selected heldout source IDs differ from scored model IDs")
    if len({r["task_id"] for r in model["cases"]}) != 8:
        raise ValueError("duplicate model task")
    source_to_task = {ident.removeprefix("musique-train-"):ident for ident in by_task}
    if len(source_to_task)!=8:
        raise ValueError("non-unique source identity")

    original = {}
    for line in original_source.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") in source_to_task:
            if row["id"] in original:
                raise ValueError("duplicated physical MuSiQue source ID")
            original[row["id"]] = row
    if set(original) != set(source_to_task):
        raise ValueError("missing frozen MuSiQue annotation rows")

    per_case = []
    by_model = {x["task_id"]:x for x in model["cases"]}
    for raw_id, source in original.items():
        task_id = source_to_task[raw_id]
        frozen = by_task[task_id]["eval_task"]
        if (frozen["files"] != _full_files(source)
            or frozen["question"].split("Question: ",1)[-1] != source["question"]
            or str(frozen["expected_answer"]) != str(source["answer"])):
            raise ValueError("physical source / frozen 20-paragraph workspace drift")
        outcome = by_model[task_id]
        stats = supporting_path_stats(source, outcome)
        per_case.append({
            "task_id":task_id,"hop_count":by_task[task_id]["hop_count"],
            **stats,
        })
    per_case.sort(key=lambda x:x["task_id"])
    support_fully_visited = sum(x["all_support_paths_visited"] for x in per_case)
    misses = sum(not x["all_support_paths_visited"] for x in per_case)
    success = sum(x["correct"] for x in per_case)
    total_recall = sum(x["support_path_recall_upper_bound"] for x in per_case)/8
    return {
        "protocol":PROTOCOL,
        "model_run_protocol":MODEL_PROTOCOL,
        "model":"qwen3.5:9b-q4_K_M",
        "model_digest":EXPECTED_MODEL_DIGEST,
        "source":"bdsaglam/musique TRAIN only",
        "revision":SOURCE_REV,
        "original_source_sha256":SOURCE_LFS_SHA256,
        "n_tasks":8,
        "correct":success,
        "support_fully_visited_tasks":support_fully_visited,
        "support_not_fully_visited_tasks":misses,
        "mean_support_path_recall_upper_bound":round(total_recall,4),
        "wrong_but_all_support_paths_visited":sum(x["all_support_paths_visited"]
                                                   and not x["correct"] for x in per_case),
        "wrong_and_missing_support_path":sum(not x["all_support_paths_visited"]
                                              and not x["correct"] for x in per_case),
        "no_training":True,
        "no_extra_model_calls":True,
        "not_GAIA":True,
        "not_sufficient_evidence_proof":True,
        "warning":"Source support annotations may be incomplete; a visited path does not prove relevant text was delivered under 2000-char tool observation truncation. Analysis on one small TRAIN-origin holdout cohort only.",
        "cases":per_case,
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("source","tasks","source-report","model-result","output"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("never overwrite this posthoc audit")
    result=audit(a.source,a.tasks,a.source_report,a.model_result)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print("QWEN9B_MUSIQUE_OTHER8_SUPPORT_PATH_AUDIT",
          json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
