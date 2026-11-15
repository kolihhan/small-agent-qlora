"""Reconstruct source-complete MuSiQue TRAIN holdout workspaces for real agent rollouts.

Use physical 20-paragraph TRAIN source, not teacher-forced oracle actions.
Never count this source-only preparation as a GAIA/model improvement.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

SOURCE_REV = "667064b514d22deecae00a3dccbd92bccf88869e"
SOURCE_LFS_SHA256 = "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
EXPECTED_HOLDOUT = 24
READER_PATH = re.compile(r"^passages/(?:index|p\d+)\.txt$")


def _read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _observed_reads(row: dict) -> dict[str, str]:
    """Inspect read observations for provenance only, never as new agent hints."""
    messages = row["messages"]
    responses = {}
    for i, message in enumerate(messages[:-1]):
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not calls:
            continue
        if message.get("role") != "assistant" or len(calls) != 1:
            raise ValueError("unexpected heldout tool-call shape")
        call = calls[0]
        fn = call["function"]
        path = fn["arguments"].get("source")
        if fn.get("name") != "read" or not isinstance(path, str) or not READER_PATH.fullmatch(path):
            raise ValueError("unexpected non-read or unsafe target")
        observed = messages[i + 1]
        if observed.get("role") != "tool" or observed.get("tool_call_id") != call["id"]:
            raise ValueError("heldout read response not paired")
        txt = observed.get("content")
        if not isinstance(txt, str) or not txt.startswith("ToolMeta: ") or "\n" not in txt:
            raise ValueError("missing exact tool metadata header")
        header, body = txt.split("\n", 1)
        meta = json.loads(header.removeprefix("ToolMeta: "))
        if meta.get("offset_chars") != 0 or not isinstance(meta.get("returned_chars"), int):
            raise ValueError("unexpected read offsets")
        if len(body) != meta["returned_chars"]:
            raise ValueError("tool observation returned length mismatch")
        if path in responses:
            raise ValueError("duplicate oracle read in frozen source")
        responses[path] = body
    if "passages/index.txt" not in responses or len(responses) < 4:
        raise ValueError("incomplete recorded MuSiQue tool path proof")
    return responses


def _full_files(source: dict) -> dict[str, str]:
    chunks = source.get("paragraphs")
    if not isinstance(chunks, list) or len(chunks) != 20:
        raise ValueError("expected all 20 original source paragraphs")
    by_index = {}
    for entry in chunks:
        ix = entry.get("idx")
        if type(ix) is not int or not 0 <= ix <= 99 or ix in by_index:
            raise ValueError("duplicate/bad original passage index")
        txt = entry.get("paragraph_text")
        if not isinstance(txt, str) or not 20 <= len(txt) <= 9000:
            raise ValueError("missing physical passage text")
        by_index[ix] = (str(entry.get("title") or ""), txt)
    files = {"passages/index.txt": "\n".join(
        f"passages/p{ix}.txt | {title}" for ix, (title, _) in sorted(by_index.items())
    )}
    for ix, (_, txt) in sorted(by_index.items()):
        files[f"passages/p{ix}.txt"] = txt
    if len(files) != 21:
        raise AssertionError("not exactly 20 paragraphs and index")
    return files


def recover(source_rows, holdout: list[dict], train: list[dict]):
    """Output independent workspaces; model is never shown previous oracle actions."""
    if len(holdout) != 35 or len(train) != 304:
        raise ValueError("v8 frozen parent count changed")
    targets = {row["task_id"]: row for row in holdout if row.get("source") == "musique-train"}
    train_ids = {row["task_id"] for row in train}
    if len(targets) != EXPECTED_HOLDOUT or train_ids & targets.keys():
        raise ValueError("MuSiQue holdout missing or leaked into training")
    id_to_task = {k.removeprefix("musique-train-"): k for k in targets}
    if len(id_to_task) != len(targets) or any(not k.startswith("musique-train-") for k in targets):
        raise ValueError("invalid MuSiQue holdout source identities")
    outstanding = set(id_to_task)
    tasks = []
    for source in source_rows:
        ident = str(source.get("id") or "")
        if ident not in outstanding:
            continue
        hold = targets[id_to_task[ident]]
        outstanding.remove(ident)
        if hold.get("verified") is not True:
            raise ValueError("source row not verified")
        if (sha256(ident.encode()).hexdigest() != hold.get("source_id_sha256")
            or hold["messages"][1]["content"].split("Question: ", 1)[-1] != source.get("question")
            or str(hold["expected_answer"]) != str(source.get("answer"))
            or hold["source"] != "musique-train"):
            raise ValueError("recovered source does not match frozen holdout parent")
        files = _full_files(source)
        for path, body in _observed_reads(hold).items():
            if files[path] != body:
                raise ValueError(f"source observation mismatch: {ident} / {path}")
        tasks.append({
            "task_id": hold["task_id"], "source": "musique-train-locked-holdout",
            "hop_count": hold["hop_count"],
            "eval_task": {
                "question": hold["messages"][1]["content"],
                "expected_answer": hold["expected_answer"],
                "required_tools": ["read"],
                "capability": "multi_hop_evidence_to_final",
                "files": files,
            },
        })
    if outstanding:
        raise ValueError(f"frozen source identities not found: {len(outstanding)}")
    tasks.sort(key=lambda x: x["task_id"])
    return tasks, {
        "protocol": "v8-musique-real-20passage-autonomous-probe-v1",
        "frozen_heldout_parents": 35, "eligible_mu_parents": len(tasks),
        "full_passages_per_task": 20, "tool_calls_teacher_forced": False,
        "source": f"bdsaglam/musique@{SOURCE_REV} TRAIN ONLY",
        "train_parents": len(train), "train_holdout_task_overlap": 0,
        "scoring": "autonomous AgentRuntime max12 via existing evaluate_policy_tasks; no model inference here",
        "not_GAIA": True, "not_model_evaluation": True,
        "provenance": "All source passages independently reconstructed; all recorded read observations exact-match physical source; no oracle actions shown",
        "limits": "TRAIN-origin holdout with annotation-derived questions, not unseen source family; needs physically verified v8 checkpoint and matched model inference",
    }


def build(train_path: Path, holdout_path: Path, source_path: Path, audit_path: Path):
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    if (sha256(train_path.read_bytes()).hexdigest() != report.get("train_sha256")
        or sha256(holdout_path.read_bytes()).hexdigest() != report.get("validation_sha256")
        or report.get("train_parents") != 304
        or report.get("validation_parents") != 35
        or report.get("no_gaia_training") is not True):
        raise ValueError("source/holdout SHA frozen validation failed")
    if sha256(source_path.read_bytes()).hexdigest() != SOURCE_LFS_SHA256:
        raise ValueError("source MuSiQue LFS exact SHA changed")
    tasks, proof = recover(_read_rows(source_path), list(_read_rows(holdout_path)), list(_read_rows(train_path)))
    proof["input_validation_sha256"] = report["validation_sha256"]
    proof["input_training_sha256"] = report["train_sha256"]
    proof["original_source_sha256"] = SOURCE_LFS_SHA256
    proof["task_manifest_sha256"] = sha256("\n".join(row["task_id"] for row in tasks).encode()).hexdigest()
    return tasks, proof


def main():
    p = argparse.ArgumentParser()
    for name in ("train", "holdout", "source", "audit", "tasks", "report"):
        p.add_argument(f"--{name}", required=True, type=Path)
    a = p.parse_args()
    if a.tasks.exists() or a.report.exists():
        raise FileExistsError("frozen evaluation artifacts must not be overwritten")
    tasks, proof = build(a.train, a.holdout, a.source, a.audit)
    a.tasks.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in tasks), encoding="utf-8")
    a.report.write_text(json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("V8_FULL_AUTONOMOUS_MUSIQUE_ENVIRONMENT_READY", json.dumps(proof, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
