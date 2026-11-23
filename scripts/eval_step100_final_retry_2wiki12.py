"""Frozen cross-family 2WikiMultiHopQA DEV12: original Step100 vs existing opt-in final_text_retry.

The *only* intervention is one extra no-tools final response when the model
emits textual <tool_call> markup on the last step. Same source, Qwen/adapter,
four tools, max_steps=12, context, search providers, score and temperature.
No task texts, answers, tool observations or full traces are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files

SOURCE_SHA = "48b9bdc69654dc580fda5f935a48b88cb89f11887587310af60d406c8d0111a6"
SOURCE_REV = "16852fde9d85cba158cf7e6517e7a3f9415a28c0"
SEED = "independent-2wiki-dev12-step100-final-retry-20261123-frozen-v1"
PROTOCOL = "QWEN-STEP100-2WIKI-DEV12-FINAL-TEXT-RETRY-V1"
N, SHARDS, PER_SHARD = 12, 3, 4
ARMS = ("control", "retry")


def _valid(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    if not (isinstance(row.get("_id"), str) and row["_id"]
            and isinstance(row.get("question"), str) and row["question"].strip()
            and isinstance(row.get("answer"), str) and row["answer"].strip()):
        return False
    context = row.get("context")
    if not isinstance(context, list) or not 4 <= len(context) <= 20:
        return False
    for item in context:
        if not isinstance(item, list) or len(item) != 2:
            return False
        title, sentences = item
        if not isinstance(title, str) or not isinstance(sentences, list):
            return False
        if not 1 <= len(sentences) <= 100 or not all(isinstance(x, str) for x in sentences):
            return False
        if not 20 <= sum(len(x) for x in sentences) <= 9000:
            return False
    return True


def load_frozen(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("2Wiki DEV source SHA mismatch")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 12576:
        raise ValueError("2Wiki DEV source count mismatch")
    ids = set()
    pool = []
    for row in rows:
        if not _valid(row):
            continue
        ident = row["_id"]
        if ident in ids:
            raise ValueError("Duplicate source ID")
        ids.add(ident)
        pool.append((hashlib.sha256((SEED + ":" + ident).encode()).hexdigest(), ident, row))
    if len(pool) < N:
        raise ValueError("Too few usable source rows")
    selected = [r for _, _, r in sorted(pool)[:N]]
    selected.sort(key=lambda r: hashlib.sha256((SEED + ":order:" + r["_id"]).encode()).hexdigest())
    manifest = hashlib.sha256("\n".join(r["_id"] for r in selected).encode()).hexdigest()
    return selected, manifest


def materialize(row: dict) -> dict:
    files = {}
    index = []
    for i, (title, sentences) in enumerate(row["context"]):
        path = f"passages/p{i}.txt"
        text = f"Title: {title}\n" + "\n".join(sentences)
        files[path] = text
        index.append(f"{path} | {title}")
    files["passages/index.txt"] = "\n".join(index)
    return {
        "question": (
            "Answer using ONLY the provided workspace passage files. "
            "Read passages/index.txt to find the relevant sources. "
            "Do not search the internet for a benchmark answer. "
            "Question: " + row["question"]
        ),
        "files": files,
        "expected_answer": row["answer"],
    }


def run(path: Path, shard: int, arm: str, base_url: str, workroot: Path) -> dict:
    if shard not in range(SHARDS) or arm not in ARMS:
        raise ValueError("Unknown frozen shard/arm")
    selected, manifest = load_frozen(path)
    model = OpenAICompatModel(model="qwen3.5-9b", base_url=base_url, max_new_tokens=512)
    cases = []
    for row in selected[shard * PER_SHARD:(shard + 1) * PER_SHARD]:
        task = materialize(row)
        ident = row["_id"]
        workspace = workroot / hashlib.sha256(ident.encode()).hexdigest()[:18]
        _materialize_files(task, workspace)
        agent = AgentRuntime(model, default_tools(), max_steps=12, final_text_retry=(arm == "retry"))
        started = time.monotonic()
        result = agent.run(task["question"], workspace)
        cases.append({
            "id_sha256": hashlib.sha256((SEED + ":" + ident).encode()).hexdigest(),
            "correct": bool(result.completed and gaia_score(result.answer, task["expected_answer"])),
            "completed": bool(result.completed),
            "stop_reason": result.stop_reason,
            "steps": result.metrics.steps,
            "tool_calls": result.metrics.tool_calls,
            "tool_errors": result.metrics.tool_errors,
            "duplicate_calls_blocked": result.metrics.duplicate_calls_blocked,
            "final_text_retries": result.metrics.final_text_retries,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
    return {
        "protocol": PROTOCOL, "arm": arm, "shard": shard, "n": PER_SHARD,
        "source_sha": SOURCE_SHA, "source_rev": SOURCE_REV,
        "selection_manifest": manifest,
        "base_adapter": "physically_audited_qwen3.5_9b_v8_step100",
        "max_steps": 12, "max_new_tokens": 512,
        "tool_names": ["search", "read", "inspect", "python"],
        "only_variable": "final_text_retry_opt_in",
        "no_train_or_gold_in_decisions": True,
        "no_task_or_answer_content_exported": True,
        "cases": cases,
    }


def score(root: Path) -> dict:
    all_pairs = []
    seen = set()
    manifest = None
    for shard in range(SHARDS):
        arms = {}
        for arm in ARMS:
            obj = json.loads((root / f"shard-{shard}-{arm}.json").read_text())
            if (obj.get("protocol") != PROTOCOL or obj.get("n") != PER_SHARD
                or obj.get("arm") != arm or obj.get("shard") != shard
                or obj.get("source_sha") != SOURCE_SHA or obj.get("source_rev") != SOURCE_REV
                or obj.get("base_adapter") != "physically_audited_qwen3.5_9b_v8_step100"
                or obj.get("only_variable") != "final_text_retry_opt_in"
                or obj.get("tool_names") != ["search", "read", "inspect", "python"]
                or obj.get("max_steps") != 12 or obj.get("max_new_tokens") != 512
                or obj.get("no_train_or_gold_in_decisions") is not True
                or obj.get("no_task_or_answer_content_exported") is not True):
                raise ValueError("Frozen evaluation protocol drift")
            if manifest is None:
                manifest = obj["selection_manifest"]
            if manifest != obj["selection_manifest"]:
                raise ValueError("Selection drift")
            cases = {x["id_sha256"]: x for x in obj["cases"]}
            if len(cases) != PER_SHARD:
                raise ValueError("Missing/duplicate cases")
            for item in cases.values():
                if (type(item["correct"]) is not bool or type(item["completed"]) is not bool
                    or item["steps"] not in range(1, 13)
                    or item["tool_calls"] < 0 or item["tool_errors"] < 0
                    or item["final_text_retries"] not in (0, 1)
                    or item["elapsed_seconds"] < 0):
                    raise ValueError("Invalid summary row")
                if arm == "control" and item["final_text_retries"] != 0:
                    raise ValueError("Control used intervention")
            arms[arm] = cases
        if arms["control"].keys() != arms["retry"].keys() or seen.intersection(arms["control"]):
            raise ValueError("Non-paired or repeated cases")
        seen.update(arms["control"])
        all_pairs.extend((arms["control"][k], arms["retry"][k]) for k in sorted(arms["control"]))
    if len(seen) != N:
        raise ValueError("Incomplete 12-task pair")
    base = sum(a["correct"] for a,b in all_pairs)
    candidate = sum(b["correct"] for a,b in all_pairs)
    rescues = sum(not a["correct"] and b["correct"] for a,b in all_pairs)
    regressions = sum(a["correct"] and not b["correct"] for a,b in all_pairs)
    retries = sum(b["final_text_retries"] for a,b in all_pairs)
    base_invalid = sum(a["stop_reason"] == "invalid_final" for a,b in all_pairs)
    retry_invalid = sum(b["stop_reason"] == "invalid_final" for a,b in all_pairs)
    time0 = sum(a["elapsed_seconds"] for a,b in all_pairs)
    time1 = sum(b["elapsed_seconds"] for a,b in all_pairs)
    gate = {
        "net_gain_at_least_2": candidate - base >= 2,
        "regressions_at_most_1": regressions <= 1,
        "at_least_one_retry_triggered": retries >= 1,
        "invalid_final_does_not_increase": retry_invalid <= base_invalid,
        "latency_at_most_125pct": time1 <= 1.25 * time0,
    }
    return {
        "protocol": PROTOCOL, "n": N, "source": "2WikiMultiHopQA DEV (new source family, may exist in public LLM pretraining)",
        "control_correct": base, "retry_correct": candidate,
        "rescues": rescues, "regressions": regressions, "final_retry_events": retries,
        "control_invalid_final": base_invalid, "retry_invalid_final": retry_invalid,
        "control_calls": sum(a["tool_calls"] for a,b in all_pairs),
        "retry_calls": sum(b["tool_calls"] for a,b in all_pairs),
        "control_tool_errors": sum(a["tool_errors"] for a,b in all_pairs),
        "retry_tool_errors": sum(b["tool_errors"] for a,b in all_pairs),
        "control_elapsed_seconds": round(time0, 2), "retry_elapsed_seconds": round(time1, 2),
        "gate": gate, "gate_pass": all(gate.values()), "no_gold_in_policy": True,
        "not_GAIA_score": True,
        "limitation": "Small single cross-family test; final-retry only handles textual tool-call markers on the last step. No GAIA improvement claim. Never retune on these 12 IDs.",
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--work-root",type=Path)
    p.add_argument("--base-url",default="http://127.0.0.1:8080")
    p.add_argument("--evidence-dir",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("Refusing overwrite of experimental evidence")
    result=(score(a.evidence_dir) if a.evidence_dir else
            run(a.source,a.shard,a.arm,a.base_url,a.work_root))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    if a.evidence_dir:
        print("2WIKI_STEP100_FINAL_RETRY_OFFICIAL",json.dumps(result,sort_keys=True))
    else:
        print("2WIKI_STEP100_FINAL_RETRY_SHARD",a.shard,a.arm,
              "correct",sum(x["correct"] for x in result["cases"]),"of",PER_SHARD)


if __name__ == "__main__":
    main()
