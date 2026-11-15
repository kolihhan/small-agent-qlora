"""Independent, source-frozen MuSiQue8 Qwen9B read-path/answer diagnostic.

No training or prompt tuning. Original 8 scored 0/8, but contained no read
paths/answers. Freeze 8 OTHER untouched tasks before decoding; record evidence
coverage, answer, and stop reason without model-serving changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files

from scripts.eval_qwen9b_musique8_cpu_baseline import (
    MODEL, EXPECTED_MODEL_DIGEST, EXPECTED_MANIFEST,
    freeze as freeze_previous, ollama_model_digest,
)

PROTOCOL = "base-qwen9b-multihop-other8-read-path-failure-diagnostic-v1"
SEED = "mu-train24-other8-trace-20261115-sha-frozen-v1"


def freeze_other(rows: list[dict], source_report: dict) -> list[dict]:
    previously_scored = {r["task_id"] for r in freeze_previous(rows, source_report)}
    selected: list[dict] = []
    for hop in (3, 4):
        fresh = [r for r in rows if r["hop_count"] == hop
                 and r["task_id"] not in previously_scored]
        if len(fresh) != 8:
            raise ValueError("previously used eight cases are not excluded")
        selected.extend(sorted(
            fresh,
            key=lambda r: sha256(f"{SEED}:{r['task_id']}".encode("utf-8")).hexdigest()
        )[:4])
    if len(selected) != 8 or len({x["task_id"] for x in selected}) != 8:
        raise ValueError("non-independent failure diagnostic cohort")
    return selected


def _path_summary(result, task: dict) -> dict:
    """Collect observed paths ONLY from runtime trace, never from target gold."""
    read_paths = []
    tool_names = []
    for event in result.trace:
        if event.kind != "tool_call":
            continue
        name = str(event.data.get("name") or "")
        tool_names.append(name)
        if name == "read":
            raw = event.data.get("arguments", {}).get("source")
            if isinstance(raw, str):
                read_paths.append(raw)
    names = {}
    for line in task["files"]["passages/index.txt"].splitlines():
        if " | " in line:
            path, title = line.split(" | ", 1)
            names[path] = title
    pass_paths = {p for p in read_paths if p in names and p != "passages/index.txt"}
    expected = str(task["expected_answer"]).strip()
    literal_present = any(expected.casefold() in task["files"][p].casefold()
                          for p in pass_paths) if expected else False
    return {
        "first_tool": tool_names[0] if tool_names else None,
        "first_read_path": read_paths[0] if read_paths else None,
        "index_read": "passages/index.txt" in read_paths,
        "read_paths": read_paths,
        "read_titles": [names.get(p, "[unrecognized path]") for p in read_paths],
        "unique_passage_count": len(pass_paths),
        "reference_literal_in_read_passages": literal_present,
        "tool_names": tool_names,
    }


def evaluate(rows: list[dict], shard: int, root: Path, base_url: str) -> dict:
    if type(shard) is not int or shard not in range(4):
        raise ValueError("four disjoint diagnostic shards required")
    digest = ollama_model_digest(base_url, MODEL)
    chosen = rows[shard*2:shard*2+2]
    if len(chosen) != 2:
        raise ValueError("incomplete diagnostic shard")
    model = OllamaModel(model=MODEL, base_url=base_url, timeout_s=900,
                        max_new_tokens=512, enable_thinking=False)
    runtime = AgentRuntime(model, default_tools(), max_steps=12)
    outcomes = []
    for item in chosen:
        task = item["eval_task"]
        workspace = root / item["task_id"]
        _materialize_files(task, workspace)
        begin = monotonic()
        result = runtime.run(task["question"], workspace)
        info = _path_summary(result, task)
        outcome = {
            "task_id": item["task_id"], "hop_count": item["hop_count"],
            "correct": bool(result.completed and gaia_score(result.answer, task["expected_answer"])),
            "completed": bool(result.completed), "stop_reason": str(result.stop_reason),
            "answer": result.answer, "expected_answer": str(task["expected_answer"]),
            "agent_steps": result.metrics.steps, "tool_calls": result.metrics.tool_calls,
            "tool_errors": result.metrics.tool_errors,
            "duplicate_calls_blocked": result.metrics.duplicate_calls_blocked,
            "latency_s": round(monotonic()-begin, 1),
            **info,
        }
        outcomes.append(outcome)
        print("MU_OTHER8_BLIND_DIAG",json.dumps({
            "id":item["task_id"],"correct":outcome["correct"],
            "stop":outcome["stop_reason"],"unique_passages":info["unique_passage_count"],
            "read_index":info["index_read"],"tools":outcome["tool_calls"]
        },sort_keys=True),flush=True)
    return {
        "protocol": PROTOCOL, "shard": shard, "model": MODEL, "model_digest": digest,
        "physical_source_manifest": EXPECTED_MANIFEST,
        "task_selection_hash": sha256("\n".join(x["task_id"] for x in rows).encode()).hexdigest(),
        "max_steps": 12, "max_new_tokens": 512,
        "no_training": True, "not_GAIA": True, "not_v8": True,
        "cases": outcomes,
    }


def aggregate(shards: list[dict]) -> dict:
    if (len(shards) != 4 or sorted(s.get("shard") for s in shards) != list(range(4))
        or len({s.get("task_selection_hash") for s in shards}) != 1
        or any(s.get("protocol") != PROTOCOL
               or s.get("model_digest") != EXPECTED_MODEL_DIGEST
               or s.get("physical_source_manifest") != EXPECTED_MANIFEST
               or len(s.get("cases", [])) != 2 for s in shards)):
        raise ValueError("incomplete independent Qwen diagnostic shards")
    cases = [c for shard in sorted(shards,key=lambda s:s["shard"]) for c in shard["cases"]]
    if (len(cases) != 8 or len({c["task_id"] for c in cases}) != 8
        or Counter(c["hop_count"] for c in cases) != Counter({3:4,4:4})):
        raise ValueError("unexpected eight unique three/four hop diagnostic questions")
    return {
        "protocol": PROTOCOL, "total": 8,
        "correct": sum(c["correct"] for c in cases),
        "completed": sum(c["completed"] for c in cases),
        "total_tools": sum(c["tool_calls"] for c in cases),
        "tool_errors": sum(c["tool_errors"] for c in cases),
        "read_index": sum(c["index_read"] for c in cases),
        "unique_read_passages": sum(c["unique_passage_count"] for c in cases),
        "reference_literal_in_read": sum(c["reference_literal_in_read_passages"] for c in cases),
        "stop_reasons": dict(Counter(c["stop_reason"] for c in cases)),
        "by_hop": {str(hop):{"total":4,"correct":sum(c["correct"] for c in cases
                                                   if c["hop_count"]==hop)} for hop in (3,4)},
        "promote": False, "model_improvement_claim": False,
        "limits": "TRAIN-derived source-selected 8 *different* from previous 8; no actual v8 adapter; literal-in-read is crude evidence probe, NOT proof of reasoning sufficiency",
        "cases": cases,
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("tasks","source-report","output"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--work-root",type=Path,default=Path("work/other8"))
    p.add_argument("--base-url",default="http://127.0.0.1:11434")
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwriting existing run")
    if args.inputs:
        result=aggregate([json.loads(pth.read_text(encoding="utf-8")) for pth in args.inputs])
        print("MU_OTHER8_DIAGNOSTIC_FINAL",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        rows=[json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
        previous_report=json.loads(args.source_report.read_text(encoding="utf-8"))
        selected=freeze_other(rows, previous_report)
        result=evaluate(selected, args.shard, args.work_root, args.base_url)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
