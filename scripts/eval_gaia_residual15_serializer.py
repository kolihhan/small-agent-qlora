"""Prospective GAIA residual15: strict, same-trajectory post-answer Qwen serializer.

GAIA validation 2023: reserve ALL previously used Diagnostic25, Evaluation100,
Shadow25. Evaluate the 15 remaining identities once only, no GAIA gold/reference
available to agent/formatter, and no training or stronger model. Original Qwen
3.5-9B + exact v7 LoRA adapter performs autonomous task; *same running v7*
model serializes ONLY its completed final answer using a literal copy of the
Hotpot16 pre-registered rule. Correctness is scored after inference.

This is public GAIA validation, not hidden; prior untracked exposures possible.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AgentResult
from gaia_small_agent.benchmark import runner
from gaia_small_agent.benchmark.gaia100 import (
    GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation,
    select_gaia_partition,
)
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools
from scripts.eval_hotpot_serializer_new16 import (
    SERIALIZER_SHA, serialize,
)

PROTOCOL="GAIA-RESIDUAL15-V7-SAME-TRAJECTORY-HOTPOT-LOCKED-SERIALIZER-V1"
N=15
SHARDS=3
MODEL="qwen3.5-9b"
# Exactly same original base & v7 adapter server as prior GAIA100 run 43865621132:
BASE_SHA="8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
V7_SHA="475f5fb59a553390a0dde9425d36b42ff85a0345bdd31b8465c22fd4d2bc02e2"
EXPECTED_SERIALIZER_SHA="a1f844029179ab91e9ff0ebd0b9b8a6293801c686e4080921edd5b3a82ef60a3"
GATE={"net_rescues":2,"max_regressions":0,"min_levels":2}


def selected_residual(dataset: list[dict]) -> list[dict]:
    # All original 3 protected partitions have pinned existing identity logic.
    used={}
    for name,n in (("diagnostic",25),("shadow",25),("evaluation",100)):
        rows=select_gaia_partition(dataset,partition=name,seed=LOCAL_PROTOCOL_SEED)
        if len(rows)!=n or len({str(r["task_id"]) for r in rows})!=n:
            raise RuntimeError(f"{name} GAIA partition changed")
        used[name]={str(r["task_id"]) for r in rows}
    if any(used[a]&used[b] for a,b in (
            ("diagnostic","shadow"),("diagnostic","evaluation"),("shadow","evaluation"))):
        raise RuntimeError("Protected GAIA partitions not disjoint")
    remaining=[r for r in dataset if str(r["task_id"]) not in set.union(*used.values())]
    if (len(dataset)!=165 or len(remaining)!=N or
        len({r["task_id"] for r in remaining})!=N):
        raise RuntimeError("Unexpected GAIA validation population/residual")
    return sorted(remaining,key=lambda r:sha256(
        (LOCAL_PROTOCOL_SEED+":residual15:"+str(r["task_id"])).encode()).hexdigest())


def freeze(rows:list[dict]) -> dict:
    if SERIALIZER_SHA!=EXPECTED_SERIALIZER_SHA:
        raise RuntimeError("Hotpot16 preregistered formatter changed after score")
    subset=selected_residual(rows)
    ids=[str(r["task_id"]) for r in subset]
    raw="\n".join(ids).encode()
    return {
        "status":"SOURCE_LOCKED_BEFORE_ANY_MODEL_INFERENCE",
        "protocol":PROTOCOL,"source_revision":GAIA_REVISION,
        "partition":"validation_residual15_excluding_diagnostic25_shadow25_evaluation100",
        "n":N,"shards":SHARDS,
        "task_id_sequence_sha256":sha256(raw).hexdigest(),
        "hashed_tasks":[sha256(x.encode()).hexdigest() for x in ids],
        "level_distribution":dict(sorted(Counter(str(r["Level"]) for r in subset).items())),
        "serializer_rule_sha":SERIALIZER_SHA,
        "same_base_and_v7_adapter":True,"original_model_runs_per_case":1,
        "extra_formatter_calls_only_for_original_completed_nonempty":True,
        "no_reference_in_model_prompt":True,
        "no_GAIA_train_or_finetune":True,
        "source_not_hidden_and_global_historical_exposure_not_proven":True,
        "preregistered_gate":GATE,
    }


class SameTrajectoryRuntime:
    """run() returns only serialized answer; original frozen by capture first."""
    def __init__(self):
        self.original=AgentRuntime(
            OpenAICompatModel(model=MODEL,max_new_tokens=512),
            default_tools(),max_steps=12)
        self.formatter=OpenAICompatModel(model=MODEL,
                                        max_new_tokens=128)
        self.captured=[]

    def run(self,question:str,workspace:Path)->AgentResult:
        original=self.original.run(question,workspace)
        raw=original.answer
        formatted,status=(raw,"original_not_completed")
        if original.completed and raw.strip():
            formatted,status=serialize(question,raw,self.formatter)
        self.captured.append({
            "original_answer":raw,"serialized_answer":formatted,
            "completed":original.completed,"stop_reason":original.stop_reason,
            "tool_calls":original.metrics.tool_calls,
            "tool_errors":original.metrics.tool_errors,
            "formatter_status":status,
            "extra_formatter_attempted":bool(original.completed and raw.strip()),
        })
        return AgentResult(answer=formatted,completed=original.completed,
                           stop_reason=original.stop_reason,
                           trace=original.trace,metrics=original.metrics)


def evaluate_shard(rows:list[dict],locked:dict,shard:int)->dict:
    if shard not in range(SHARDS):
        raise ValueError("Bad residual shard")
    current=freeze(rows)
    if current!=locked:
        raise RuntimeError("GAIA residual/source/formatter pre-registration changed")
    selected=selected_residual(rows)
    subset=shard_selected_rows(selected,shard_index=shard,shard_count=SHARDS)
    if len(subset)!=5:
        raise RuntimeError("Residual GAIA shard must have 5 cases")
    old_selector=runner.select_gaia_partition
    runtime=SameTrajectoryRuntime()
    try:
        with TemporaryDirectory(prefix="gaia-residual15-serializer-") as tmp:
            runner.select_gaia_partition=lambda *_args,**_kwargs:subset
            runner.run_gaia100(
                subset,lambda:runtime,Path(tmp),
                seed=LOCAL_PROTOCOL_SEED,partition="evaluation",
                dataset_revision=GAIA_REVISION)
            stored=[json.loads(x) for x in (
                Path(tmp)/"results.jsonl").read_text().splitlines() if x.strip()]
    finally:
        runner.select_gaia_partition=old_selector
    if len(stored)!=len(subset) or len(runtime.captured)!=len(subset):
        raise RuntimeError("GAIA original and treated trajectories missing")
    report=[]
    for source,graded,cap in zip(subset,stored,runtime.captured):
        if graded["task_id"]!=source["task_id"]:
            raise RuntimeError("Scorer source ordering drift")
        if graded["model_answer"]!=cap["serialized_answer"]:
            raise RuntimeError("Runner did not grade the exact new answer")
        gold=str(source["Final answer"])
        raw_correct=bool(cap["completed"] and gaia_score(cap["original_answer"],gold))
        new_correct=bool(graded["correct"])
        if new_correct != bool(cap["completed"] and gaia_score(
                cap["serialized_answer"],gold)):
            raise RuntimeError("Posthoc strict scorer disagrees with runner")
        if (graded["tool_calls"]!=cap["tool_calls"]
            or graded["tool_errors"]!=cap["tool_errors"]
            or graded["completed"]!=cap["completed"]):
            raise RuntimeError("Agent trajectory changed by formatter")
        report.append({
            "task_hash":sha256(source["task_id"].encode()).hexdigest(),
            "level":int(source["Level"]),
            "original_correct":raw_correct,
            "serialized_correct":new_correct,
            "original_completed":cap["completed"],
            "same_trace_no_tool_change":True,
            "tool_calls":cap["tool_calls"],
            "tool_errors":cap["tool_errors"],
            "stop_reason":cap["stop_reason"],
            "formatter_status":cap["formatter_status"],
            "extra_formatter_attempted":cap["extra_formatter_attempted"],
            # NO GAIA prompts, gold, raw or new candidate answers in artifact.
            "raw_hash":sha256(cap["original_answer"].encode()).hexdigest(),
            "serialized_hash":sha256(cap["serialized_answer"].encode()).hexdigest(),
        })
    return {
        "status":"completed","protocol":PROTOCOL,
        "source_hash":locked["task_id_sequence_sha256"],
        "serializer_sha":SERIALIZER_SHA,"base_sha":BASE_SHA,
        "adapter_sha":V7_SHA,"shard":shard,"n":5,
        "cases":report,
    }


def aggregate(shards:list[dict])->dict:
    if (len(shards)!=SHARDS or
        sorted(s.get("shard") for s in shards)!=list(range(SHARDS)) or
        len({s.get("source_hash") for s in shards})!=1 or
        any(s.get("status")!="completed" or s.get("protocol")!=PROTOCOL or
            s.get("serializer_sha")!=SERIALIZER_SHA or
            s.get("base_sha")!=BASE_SHA or s.get("adapter_sha")!=V7_SHA or
            s.get("n")!=5 or len(s.get("cases",[]))!=5
            for s in shards)):
        raise RuntimeError("Incomplete GAIA model/source/same-serializer matched shards")
    rows=[r for s in sorted(shards,key=lambda x:x["shard"]) for r in s["cases"]]
    if len(rows)!=N or len({r["task_hash"] for r in rows})!=N:
        raise RuntimeError("Missing/duplicate protected GAIA residual cases")
    rescue=[r for r in rows if not r["original_correct"] and r["serialized_correct"]]
    regress=[r for r in rows if r["original_correct"] and not r["serialized_correct"]]
    levels={r["level"] for r in rescue}
    positive=(len(rescue)-len(regress)>=GATE["net_rescues"]
              and len(regress)<=GATE["max_regressions"]
              and len(levels)>=GATE["min_levels"])
    return {
        "status":"COMPLETE_GAIA_PUBLIC_RESIDUAL15_SCREEN",
        "protocol":PROTOCOL,"source_sha":shards[0]["source_hash"],
        "n":N,"base_gguf_sha256":BASE_SHA,"v7_adapter_gguf_sha256":V7_SHA,
        "hotpot_frozen_formatter_sha256":SERIALIZER_SHA,
        "original_correct":sum(r["original_correct"] for r in rows),
        "serialized_correct":sum(r["serialized_correct"] for r in rows),
        "original_completed":sum(r["original_completed"] for r in rows),
        "rescues":len(rescue),"regressions":len(regress),
        "net_improvement":len(rescue)-len(regress),
        "rescues_by_level":dict(sorted(Counter(r["level"] for r in rescue).items())),
        "original_by_level":dict(sorted(Counter(r["level"] for r in rows if r["original_correct"]).items())),
        "serialized_by_level":dict(sorted(Counter(r["level"] for r in rows if r["serialized_correct"]).items())),
        "extra_formatter_calls":sum(r["extra_formatter_attempted"] for r in rows),
        "accepted_verbatim":sum(r["formatter_status"]=="accepted_verbatim" for r in rows),
        "tool_calls_original_unchanged":sum(r["tool_calls"] for r in rows),
        "tool_errors_original_unchanged":sum(r["tool_errors"] for r in rows),
        "predeclared_gate":GATE,"positive_screen_only":positive,
        "promotion_allowed":False,"not_official_GAIA100_or_hidden":True,
        "no_training_no_new_adapter":True,
        "claim_boundary":"GAIA public validation residual15 only; excluded previously reserved diagnostic25/shadow25/evaluation100, but untracked historical exposure cannot be ruled out. Frozen Hotpot16 rule applied with same original v7 base & adapter, no gold in prompt, no new reasoning, raw and new strict correctness computed against same reference after original autonomous trajectory. Zero evidence of generalization to official hidden GAIA or historical GAIA100 without further matched tests.",
        "cases":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-only",action="store_true")
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--shard",type=int)
    parser.add_argument("--inputs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.inputs:
        result=aggregate([json.loads(p.read_text()) for p in a.inputs])
        print("GAIA_RESIDUAL15_SERIALIZER_FINAL",json.dumps(
            {k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        dataset=load_validation()
        if a.source_only:
            result=freeze(dataset)
            print("GAIA_RESIDUAL15_FROZEN",result["task_id_sequence_sha256"],flush=True)
        else:
            if a.shard is None or a.frozen is None:
                parser.error("Need --frozen and --shard for model replay")
            result=evaluate_shard(dataset,json.loads(a.frozen.read_text()),a.shard)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
