"""Exact-prefix causal probe: replay only the forced final step of GAIA agent.

The original Qwen3.5-9B/v7 AgentRuntime executes the whole task ONCE. Capture
the identical final-step messages and tool schema BEFORE original final
decoding; replay the SAME messages only with llama.cpp JSON Schema constraint.
No second tool trajectory, no modified earlier inference, no prompt change.
If original never reaches step-12 terminal, no intervention occurs and case
is excluded from the terminal intervention denominator.

EXPOSED GAIA DEV8: mechanism screen, not hidden score/generalization.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time

from gaia_small_agent.agent.loop import AgentRuntime,_looks_like_textual_tool_call
from gaia_small_agent.agent.types import ModelRuntimeError
from gaia_small_agent.benchmark import runner
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION,LOCAL_PROTOCOL_SEED,load_validation
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools

from scripts.gaia_json_schema_terminal8 import (
    ConstrainedTerminalModel,FROZEN_IDS,FROZEN_SHA,MODEL,select_frozen,
)

N_SHARDS=4
PROTOCOL="GAIA8-QWEN35-V7-FIXED-PREFIX-JSON-TERMINAL-V1"


class OriginalFinalCapturer(OpenAICompatModel):
    def __init__(self):
        super().__init__(MODEL,max_new_tokens=512)
        self.original_final_input=None

    def complete_final(self,messages:list[dict],tools:list[dict]):
        if self.original_final_input is not None:
            raise RuntimeError("Expected exactly one original final-step request per GAIA task")
        self.original_final_input=(copy.deepcopy(messages),copy.deepcopy(tools))
        return super().complete_final(messages,tools)


def check_final_result(answer:str|None)->tuple[bool,str]:
    if not isinstance(answer,str) or not answer.strip():
        return False,"empty_final"
    if _looks_like_textual_tool_call(answer.strip()):
        return False,"invalid_final"
    return True,"final"


def run_case(row:dict)->dict:
    original=OriginalFinalCapturer()
    with tempfile.TemporaryDirectory(prefix="gaia-fixed-prefix-json-") as root:
        previous=runner.select_gaia_partition
        try:
            runner.select_gaia_partition=lambda *_a,**_kw:[row]
            runner.run_gaia100(
                [row],
                lambda:AgentRuntime(original,default_tools(),max_steps=12),
                Path(root),
                seed=LOCAL_PROTOCOL_SEED,partition="evaluation",dataset_revision=GAIA_REVISION,
            )
        finally:
            runner.select_gaia_partition=previous
        records=[json.loads(line) for line in (Path(root)/"results.jsonl").read_text().splitlines() if line.strip()]
    if len(records)!=1 or records[0]["task_id"]!=row["task_id"]:
        raise RuntimeError("Original model produced missing/mismatched task record")
    baseline=records[0]
    output={
        "task_hash":sha256(str(row["task_id"]).encode()).hexdigest(),
        "original":{
            "correct":bool(baseline["correct"]),"completed":bool(baseline["completed"]),
            "stop_reason":baseline["stop_reason"],
            "steps":int(baseline["steps"]),
            "tool_calls":int(baseline["tool_calls"]),
            "tool_errors":int(baseline["tool_errors"]),
        },
        "terminal_reached":original.original_final_input is not None,
    }
    if original.original_final_input is None:
        # Difference=0 by design when there is no terminal intervention.
        output["replayed"]=False
        output["json_schema_final"]={
            **output["original"],"no_intervention":True,
        }
        return output
    messages,tools=original.original_final_input
    source=json.dumps([messages,tools],ensure_ascii=False,sort_keys=True).encode()
    output["exact_prefix_sha256"]=sha256(source).hexdigest()
    if int(baseline["steps"])!=12:
        raise RuntimeError("A forced terminal capture must occur precisely at max step 12")
    constrained=ConstrainedTerminalModel()
    started=time.perf_counter()
    try:
        turn=constrained.complete_final(messages,tools)
        if turn.tool_calls:
            raise ModelRuntimeError("model_error","Constrained terminal unexpectedly returned tool calls")
        ok,stop=check_final_result(turn.content)
        correct=bool(ok and gaia_score(turn.content.strip(),str(row["Final answer"])))
        output["json_schema_final"]={
            "correct":correct,"completed":ok,"stop_reason":stop,
            "steps":12,"tool_calls":int(baseline["tool_calls"]),
            "tool_errors":int(baseline["tool_errors"]),
        }
    except ModelRuntimeError as exc:
        output["json_schema_final"]={
            "correct":False,"completed":False,
            "stop_reason":exc.stop_reason,
            "steps":12,"tool_calls":int(baseline["tool_calls"]),
            "tool_errors":int(baseline["tool_errors"]),
        }
    output["replayed"]=True
    output["terminal_only_latency_s"]=round(time.perf_counter()-started,2)
    return output


def run_shard(shard:int)->dict:
    if shard not in range(N_SHARDS):
        raise ValueError("Invalid shard number")
    selected=select_frozen(load_validation())
    rows=selected[shard*2:shard*2+2]
    if len(rows)!=2:
        raise ValueError("Expected exactly 2 frozen source questions per shard")
    output=[run_case(r) for r in rows]
    if [r["task_hash"] for r in output]!=list(FROZEN_IDS[shard*2:shard*2+2]):
        raise RuntimeError("Frozen source ID mismatch")
    return {
        "status":"complete","protocol":PROTOCOL,"shard":shard,"n":2,
        "source_revision":GAIA_REVISION,"source_ids_sha256":FROZEN_SHA,
        "model":"pinned Qwen3.5-9B v7 (unchanged)",
        "only_variable":"exact same captured forced terminal messages + JSON Schema constrained decoding",
        "cases":output,
    }


def aggregate(paths:list[Path])->dict:
    if len(paths)!=N_SHARDS:
        raise ValueError("All four physical shards mandatory")
    shards=[json.loads(p.read_text()) for p in paths]
    if sorted(s.get("shard") for s in shards)!=list(range(N_SHARDS)):
        raise ValueError("Incomplete or duplicate physical shard")
    if any(s.get("status")!="complete" or s.get("protocol")!=PROTOCOL
           or s.get("n")!=2 or s.get("source_revision")!=GAIA_REVISION
           or s.get("source_ids_sha256")!=FROZEN_SHA or len(s.get("cases",[]))!=2
           for s in shards):
        raise ValueError("Protocol/source drift")
    cases=[r for s in sorted(shards,key=lambda x:x["shard"]) for r in s["cases"]]
    if len(cases)!=8 or [r["task_hash"] for r in cases]!=list(FROZEN_IDS):
        raise ValueError("Only original preregistered eight GAIA exposed IDs allowed")
    tested=[x for x in cases if x["terminal_reached"]]
    if any(not x.get("replayed") or not x.get("exact_prefix_sha256") for x in tested):
        raise ValueError("Missing exact-message terminal replays")
    if any((x["original"]["tool_calls"]!=x["json_schema_final"]["tool_calls"] or
            x["original"]["tool_errors"]!=x["json_schema_final"]["tool_errors"])
           for x in cases):
        raise ValueError("Before-final tool history changed")
    scores={}
    for arm in ("original","json_schema_final"):
        scores[arm]={
            "strict_correct":sum(r[arm]["correct"] for r in cases),
            "completed":sum(r[arm]["completed"] for r in cases),
            "invalid_final":sum(r[arm]["stop_reason"]=="invalid_final" for r in cases),
            "timeouts":sum(r[arm]["stop_reason"]=="model_timeout" for r in cases),
            "tested_terminal_correct":sum(r[arm]["correct"] for r in tested),
        }
    rescues=sum(not r["original"]["correct"] and r["json_schema_final"]["correct"] for r in tested)
    regressions=sum(r["original"]["correct"] and not r["json_schema_final"]["correct"] for r in tested)
    return {
        "status":"SOURCE_EXPOSED_EXACT_PREFIX_CAUSAL_SCREEN_COMPLETE",
        "protocol":PROTOCOL,"source_ids_sha256":FROZEN_SHA,
        "n":len(cases),"forced_terminal_reached":len(tested),
        "nonterminal_unchanged":len(cases)-len(tested),
        "arms":scores,"terminal_rescues":rescues,
        "terminal_regressions":regressions,
        "terminal_net_correctness":rescues-regressions,
        "no_tools_rerun_for_structured_arm":True,
        "original_prompt_and_runtime_unchanged":True,
        "not_promoted":True,
        "claim_boundary":"Original exposed outcome-conditioned GAIA DEV8, not hidden or unbiased; strict causal comparison only on actual terminal-reaching tasks. No generalization or 100-case gain established.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.inputs:
        r=aggregate(a.inputs)
        print("GAIA8_FIXED_PREFIX_TERMINAL_FINAL",json.dumps(r,sort_keys=True),flush=True)
    else:
        if a.shard is None:p.error("shard or inputs required")
        r=run_shard(a.shard)
        print("GAIA8_FIXED_PREFIX_TERMINAL_SHARD",a.shard,"success",flush=True)
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__":
    main()
