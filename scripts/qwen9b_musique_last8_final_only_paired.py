"""Paired Qwen3.5-9B same-model final-answer format test on last 8 untouched MuSiQue parents.

Hypothesis from *previous* independent 8: some answers contain the correct short
entity/date but get scored wrong because of explanations. Freeze LAST 8 before
generation; compare original AgentRuntime vs one generic final-only instruction.
No model training, GAIA data, gold prompts, special-case entity extraction or
autopromotion. Source is MuSiQue TRAIN-derived and cannot prove GAIA gain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from gaia_small_agent.agent.loop import AgentRuntime, SYSTEM_PROMPT
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.eval_qwen9b_musique8_cpu_baseline import (
    MODEL, EXPECTED_MODEL_DIGEST, EXPECTED_MANIFEST, freeze as freeze_first,
    ollama_model_digest,
)
from scripts.eval_qwen9b_musique_other8_readpath import freeze_other

SEED = "mu-train24-final-eight-answer-only-paired-20261115-v1"
PROTOCOL = "qwen9b-real-3-4hop-unseen-last8-final-only-one-variable-paired-v1"
FINAL_ONLY_PROMPT = SYSTEM_PROMPT + (
    "\nFinal answer policy: When ready, respond with ONLY the shortest factual "
    "answer that the question asks for: a name, place, date, number, percentage "
    "or short entity. Do not add explanations, introductions, reasoning, "
    "markdown formatting or labels such as 'Answer:'. Keep tool decisions "
    "grounded in actual observations; never guess missing facts.\n"
)


def freeze_last(rows: list[dict], report: dict) -> list[dict]:
    first={r["task_id"] for r in freeze_first(rows,report)}
    other={r["task_id"] for r in freeze_other(rows,report)}
    if first & other or len(first)!=8 or len(other)!=8:
        raise ValueError("frozen prior 16 questions overlap or drift")
    left=[r for r in rows if r["task_id"] not in first|other]
    if (len(left)!=8 or len({r["task_id"] for r in left})!=8
        or Counter(r["hop_count"] for r in left)!=Counter({3:4,4:4})):
        raise ValueError("no independent last 8 MuSiQue source cases remain")
    left.sort(key=lambda r:sha256((SEED+":"+r["task_id"]).encode()).hexdigest())
    return left


def evaluate_arm(model, task: dict, workspace: Path, *, prompt: str | None) -> dict:
    _materialize_files(task,workspace)
    runtime=AgentRuntime(model,default_tools(),max_steps=12,system_prompt=prompt)
    started=monotonic()
    result=runtime.run(task["question"],workspace)
    observed_read=[
        str(e.data.get("arguments",{}).get("source") or "")
        for e in result.trace if e.kind=="tool_call" and e.data.get("name")=="read"
    ]
    return {
        "completed":bool(result.completed),
        "answer":result.answer,
        "stop_reason":str(result.stop_reason),
        "tool_calls":result.metrics.tool_calls,
        "tool_errors":result.metrics.tool_errors,
        "duplicate_blocks":result.metrics.duplicate_calls_blocked,
        "steps":result.metrics.steps,
        "read_paths":observed_read,
        "latency_s":round(monotonic()-started,1),
    }


def run_shard(selected: list[dict], shard: int, root: Path, base_url: str) -> dict:
    if type(shard) is not int or shard not in range(4):
        raise ValueError("four source-pinned shards required")
    digest=ollama_model_digest(base_url,MODEL)
    model=OllamaModel(model=MODEL,base_url=base_url,timeout_s=900,
                      max_new_tokens=512,enable_thinking=False)
    outcomes=[]
    for task_row in selected[2*shard:2*shard+2]:
        task=task_row["eval_task"]
        key=task_row["task_id"]
        # Two INDEPENDENT workspaces with identical source files.
        base=evaluate_arm(model,task,root/"baseline"/key,prompt=None)
        treatment=evaluate_arm(model,task,root/"treatment"/key,prompt=FINAL_ONLY_PROMPT)
        # The gold answer is accessed only AFTER both candidates have been frozen.
        gold=str(task["expected_answer"])
        for arm in (base,treatment):
            arm["exact_correct"]=bool(
                arm["completed"] and gaia_score(arm["answer"],gold)
            )
            arm["gold_literal_in_output"]=bool(gold and
                gold.casefold() in arm["answer"].casefold())
        outcomes.append({
            "task_id":key,"hop_count":task_row["hop_count"],
            "expected_answer":gold,"baseline":base,"final_only":treatment,
        })
        print("QWEN9B_LAST8_PAIRED_FROZEN",
              key,"base",base["exact_correct"],
              "final_only",treatment["exact_correct"],flush=True)
    return {
        "protocol":PROTOCOL,"shard":shard,"model":MODEL,"model_digest":digest,
        "task_selection_sha256":sha256("\n".join(x["task_id"] for x in selected).encode()).hexdigest(),
        "source_manifest_sha256":EXPECTED_MANIFEST,
        "max_steps":12,"max_new_tokens":512,
        "treatment":"single generic system prompt addition: answer-only finalization",
        "no_training":True,"not_GAIA":True,"not_v8":True,
        "cases":outcomes,
    }


def aggregate(shards: list[dict], *, digest: str=EXPECTED_MODEL_DIGEST) -> dict:
    if (len(shards)!=4 or sorted(s.get("shard") for s in shards)!=list(range(4))
        or len({s.get("task_selection_sha256") for s in shards})!=1
        or any(s.get("protocol")!=PROTOCOL or s.get("model_digest")!=digest
               or s.get("source_manifest_sha256")!=EXPECTED_MANIFEST
               or len(s.get("cases",[]))!=2 for s in shards)):
        raise ValueError("incomplete source-pinned four paired shards")
    cases=[x for shard in sorted(shards,key=lambda s:s["shard"]) for x in shard["cases"]]
    if (len(cases)!=8 or len({x["task_id"] for x in cases})!=8
        or Counter(x["hop_count"] for x in cases)!=Counter({3:4,4:4})):
        raise ValueError("not eight distinct source-heldout cases")
    base=sum(c["baseline"]["exact_correct"] for c in cases)
    new=sum(c["final_only"]["exact_correct"] for c in cases)
    rescues=[c for c in cases if not c["baseline"]["exact_correct"] and c["final_only"]["exact_correct"]]
    regressions=[c for c in cases if c["baseline"]["exact_correct"] and not c["final_only"]["exact_correct"]]
    if new-base!=len(rescues)-len(regressions):
        raise AssertionError("paired answer success accounting mismatch")
    gains_hops={c["hop_count"] for c in rescues}
    return {
        "protocol":PROTOCOL,"model":MODEL,"model_digest":digest,
        "total":8,"baseline_exact":base,"final_only_exact":new,
        "rescues":len(rescues),"regressions":len(regressions),
        "rescued_hops":sorted(gains_hops),
        "baseline_finals":sum(c["baseline"]["completed"] for c in cases),
        "final_only_finals":sum(c["final_only"]["completed"] for c in cases),
        "baseline_tools":sum(c["baseline"]["tool_calls"] for c in cases),
        "final_only_tools":sum(c["final_only"]["tool_calls"] for c in cases),
        "baseline_errors":sum(c["baseline"]["tool_errors"] for c in cases),
        "final_only_errors":sum(c["final_only"]["tool_errors"] for c in cases),
        "baseline_literal_in_answer":sum(c["baseline"]["gold_literal_in_output"] for c in cases),
        "final_only_literal_in_answer":sum(c["final_only"]["gold_literal_in_output"] for c in cases),
        "baseline_latency_s":round(sum(c["baseline"]["latency_s"] for c in cases),1),
        "final_only_latency_s":round(sum(c["final_only"]["latency_s"] for c in cases),1),
        "by_hop":{str(h):{
            "total":4,
            "base_correct":sum(c["baseline"]["exact_correct"] for c in cases if c["hop_count"]==h),
            "final_only_correct":sum(c["final_only"]["exact_correct"] for c in cases if c["hop_count"]==h),
        } for h in (3,4)},
        "positive_screen_only":new-base>=2 and len(regressions)==0 and gains_hops=={3,4},
        "promotion_allowed":False,
        "not_GAIA_or_matched_v8":True,
        "caveat":"Only 8 previously unscored MuSiQue TRAIN-origin source parents, strict exact answer scoring; a positive signal still requires disjoint source and protected GAIA end-to-end verification. Tool behavior may shift under a prompt change.",
        "cases":cases,
    }


def main():
    p=argparse.ArgumentParser()
    for k in ("tasks","source-report","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--work-root",type=Path,default=Path("work/last8-paired"))
    p.add_argument("--base-url",default="http://127.0.0.1:11434")
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("do not overwrite frozen paired experiment")
    if a.inputs:
        result=aggregate([json.loads(f.read_text()) for f in a.inputs])
        print("QWEN9B_LAST8_ANSWER_ONLY_FINAL",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        rows=[json.loads(line) for line in a.tasks.read_text(encoding="utf8").splitlines() if line.strip()]
        report=json.loads(a.source_report.read_text(encoding="utf8"))
        selected=freeze_last(rows,report)
        result=run_shard(selected,a.shard,a.work_root,a.base_url)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
