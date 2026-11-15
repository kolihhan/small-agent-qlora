"""8 source-frozen MuSiQue TRAIN holdout *autonomous* Qwen9B CPU baseline.

No training, no oracle replay, no GAIA data, no v8 / paired checkpoint claim.
Use actual Read/Search/Inspect/Python tools, with 20 original physical files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files

MODEL = "qwen3.5:9b-q4_K_M"
EXPECTED_MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE_SHA = "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
EXPECTED_MANIFEST = "2767ec507540cabe91af5feb58f76c15669b10ac8d64030033347ae3733018fe"
SEED = "qwen3.5-9b-musique8-frozen-autonomous-20261115"
PROTOCOL = "base-qwen9b-ollama-multihop8-local-workspace-cpu-v1"


def freeze(rows: list[dict], report: dict) -> list[dict]:
    """Select four 3-hop and four 4-hop parents before any predictions."""
    if (len(rows) != 24 or report.get("eligible_mu_parents") != 24
        or report.get("original_source_sha256") != SOURCE_SHA
        or report.get("task_manifest_sha256") != EXPECTED_MANIFEST
        or report.get("tool_calls_teacher_forced") is not False
        or sha256("\n".join(x["task_id"] for x in rows).encode()).hexdigest() != EXPECTED_MANIFEST):
        raise ValueError("physical MuSiQue24 heldout source/report identity changed")
    selected = []
    for hop in (3, 4):
        pool = [x for x in rows if x["hop_count"] == hop]
        if len(pool) != 12:
            raise ValueError("expected 12 held-out parents of each hop count")
        for row in pool:
            task = row.get("eval_task")
            if (not isinstance(task, dict) or len(task.get("files", {})) != 21
                or len(task.get("question", "")) < 25
                or "messages" in task or "training_action_indices" in task
                or not task.get("expected_answer")):
                raise ValueError("partial workspace or oracle-leaking parent")
        selected.extend(sorted(pool,key=lambda r:sha256(
            f"{SEED}:{r['task_id']}".encode()).hexdigest())[:4])
    if len(selected)!=8 or len({x["task_id"] for x in selected})!=8:
        raise ValueError("bad frozen 8 task sample")
    return selected


def ollama_model_digest(base_url: str, model: str) -> str:
    import requests
    response=requests.get(base_url.rstrip("/")+"/api/tags",timeout=15)
    response.raise_for_status()
    rows=response.json().get("models") or []
    match=[r["digest"] for r in rows if r.get("name")==model or r.get("model")==model]
    if len(match)!=1 or match[0]!=EXPECTED_MODEL_DIGEST:
        raise ValueError("Ollama source model digest differs from pinned original Qwen3.5-9B")
    return match[0]


def evaluate(rows: list[dict], *, shard: int, root: Path, base_url: str, model_digest: str) -> dict:
    if type(shard) is not int or shard not in range(4):
        raise ValueError("four unique 2-task source-frozen shards only")
    if model_digest != EXPECTED_MODEL_DIGEST:
        raise ValueError("source model checkpoint digest drift")
    chosen = rows[2*shard:2*shard+2]
    if len(chosen)!=2:
        raise ValueError("incomplete shard")
    model=OllamaModel(model=MODEL,base_url=base_url,timeout_s=900,max_new_tokens=512,enable_thinking=False)
    runtime=AgentRuntime(model,default_tools(),max_steps=12)
    cases=[]
    for row in chosen:
        task=row["eval_task"]
        workspace=root/row["task_id"]
        _materialize_files(task,workspace)
        started=time.monotonic()
        result=runtime.run(task["question"],workspace)
        cases.append({
            "task_id":row["task_id"],"hop_count":row["hop_count"],
            "completed":bool(result.completed),
            "correct":bool(result.completed and gaia_score(result.answer,task["expected_answer"])),
            "stop_reason":str(result.stop_reason),
            "agent_steps":result.metrics.steps,"tool_calls":result.metrics.tool_calls,
            "tool_errors":result.metrics.tool_errors,
            "duplicate_calls_blocked":result.metrics.duplicate_calls_blocked,
            "tool_names":[str(e.data.get("name")) for e in result.trace if e.kind=="tool_call"],
            "latency_s":round(time.monotonic()-started,1),
        })
        print("QWEN9B_MUSIQUE8_BASE_CLOSEDLOOP_TASK",json.dumps({
            k:v for k,v in cases[-1].items() if k not in ("tool_names",)},sort_keys=True),flush=True)
    return {"protocol":PROTOCOL,"shard":shard,"model":MODEL,"model_digest":model_digest,
            "task_selection_hash":sha256("\n".join(x["task_id"] for x in rows).encode()).hexdigest(),
            "cohort_sha256":EXPECTED_MANIFEST,
            "max_steps":12,"max_new_tokens":512,
            "runtime_backend":"Ollama CPU; not same quantization/backend as Transformers NF4 v8",
            "gaia_evaluated":False,"adapter_evaluated":False,"cases":cases}


def aggregate(shards: list[dict], *, expected_digest: str=EXPECTED_MODEL_DIGEST) -> dict:
    if (len(shards)!=4 or sorted(x.get("shard") for x in shards)!=list(range(4))
        or len({x.get("task_selection_hash") for x in shards})!=1
        or len({x.get("cohort_sha256") for x in shards})!=1
        or any(x.get("protocol")!=PROTOCOL or x.get("model_digest")!=expected_digest
               or len(x.get("cases",[]))!=2 for x in shards)):
        raise ValueError("incomplete four 2-case source shards or model drift")
    rows=[case for s in sorted(shards,key=lambda x:x["shard"]) for case in s["cases"]]
    hops=Counter(x["hop_count"] for x in rows)
    if len(rows)!=8 or len({x["task_id"] for x in rows})!=8 or hops!={3:4,4:4}:
        raise ValueError("non-unique 4+4 real multitask cohort")
    return {"protocol":PROTOCOL,"model":MODEL,"model_digest":expected_digest,
            "total":8,"exact_correct":sum(x["correct"] for x in rows),
            "completed":sum(x["completed"] for x in rows),
            "tool_calls":sum(x["tool_calls"] for x in rows),
            "tool_errors":sum(x["tool_errors"] for x in rows),
            "duplicates":sum(x["duplicate_calls_blocked"] for x in rows),
            "total_latency_s":round(sum(x["latency_s"] for x in rows),1),
            "by_hop":{str(k):{"total":v,"correct":sum(x["correct"] for x in rows if x["hop_count"]==k)}
                      for k,v in sorted(hops.items())},
            "stop_reasons":dict(Counter(x["stop_reason"] for x in rows)),
            "is_matched_v8_eval":False,"is_GAIA_eval":False,"promotion_allowed":False,
            "limitations":"8-task TRAIN-origin MuSiQue diagnostic, Ollama CPU model, not Transformers NF4/GAIA and no full v8 checkpoint",
            "cases":rows}


def main():
    p=argparse.ArgumentParser()
    for k in ("tasks","source-report","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--base-url",default="http://127.0.0.1:11434")
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--work-root",default="work/musique8-base",type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("do not overwrite independent results")
    if a.inputs:
        result=aggregate([json.loads(p.read_text()) for p in a.inputs])
        print("QWEN9B_MUSIQUE8_BASE_FINAL",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        rows=[json.loads(line) for line in a.tasks.read_text().splitlines() if line.strip()]
        frozen=freeze(rows,json.loads(a.source_report.read_text()))
        digest=ollama_model_digest(a.base_url,MODEL)
        result=evaluate(frozen,shard=a.shard,root=a.work_root,base_url=a.base_url,model_digest=digest)
    a.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
