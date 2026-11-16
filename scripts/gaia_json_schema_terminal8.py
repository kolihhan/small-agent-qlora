"""Causal smoke: force a structured answer ONLY on the last GAIA agent step.

The preterminal policy, tool selection, Qwen3.5-9B GGUF and v7 LoRA,
tool budget and original prompts are IDENTICAL to the original arm.
The only intervention uses llama.cpp constrained JSON Schema for final output,
decodes a single {"answer":str} field and passes it to the unchanged GAIA
scorer. This is a selectively exposed 8-task DEV diagnostic, NOT a hidden test.

Original failure condition: the model emitted literal <tool_call> text at
the forced tool-disabled last step. Earlier last-step prompt reminder already
failed on independent GAIA15. This is DECODING, not another prompt reminder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ModelRuntimeError
from gaia_small_agent.benchmark import runner
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools

MODEL = "qwen3.5-9b"
ARMS = ("original","json_schema_final")
SHARD_COUNT = 4
N = 8
# Frozen before the new model inference: four prior invalid-final and four
# prior strictly correct, deterministic salted-hash selection from past v7 GAIA100.
# Conditional-on-prior-outcome selection = mechanism smoke, NOT prevalence estimate.
FROZEN_IDS = (
    "922c7c2eb3c66f48e1ed087f8d3c33af24271bee865cba2b08c4fe1c83256e0e",
    "fdf9b4148ac8020930a96148f27a00adb988cd9662ed58bc8b2bc282986d7709",
    "7fa94505d57024a94bef12a5bb473378ebeeea406ec881485d311dd40dd5ee44",
    "7dc365c2839eec96b0acdf27aa044464428891e47d109ff000c9cefc5ee3862c",
    "3248dafebe93c65c732b88b0bb51a13a8715df29281b492dd364169450c0e905",
    "c7901b3faa92ab790cf6eba9125750df6b19a3669e89dccb66f9d228496346ca",
    "07219a10542b2ea286d2910191c2c6c993850e4695f3b5dbb08a1a1ef1378203",
    "657c0b7a2439ab2e7396fb1cd269648ff65c45adc007b2aeea15f0852ef0430f",
)
FROZEN_SHA=sha256("|".join(FROZEN_IDS).encode()).hexdigest()
FINAL_SCHEMA={
    "type":"json_schema",
    "json_schema":{
        "name":"gaia_final_answer",
        "strict":True,
        "schema":{
            "type":"object",
            "properties":{"answer":{"type":"string","minLength":1}},
            "required":["answer"],
            "additionalProperties":False,
        },
    },
}


def strictly_decode_answer(raw:str)->str:
    """No posthoc extraction, no gold, no guess or recovery fallback."""
    try:
        payload=json.loads(raw)
    except (ValueError,TypeError) as exc:
        raise ModelRuntimeError("model_error","Constrained final did not return JSON") from exc
    if (not isinstance(payload,dict) or set(payload)!={"answer"}
        or not isinstance(payload["answer"],str) or not payload["answer"].strip()):
        raise ModelRuntimeError("model_error","Constrained final did not return a nonempty answer field")
    answer=payload["answer"].strip()
    if answer.lstrip().startswith("<tool_call>"):
        # Structured JSON syntax alone is NOT sufficient evidence of
        # a valid final answer if it smuggles an XML tool directive.
        return answer
    return answer


class ConstrainedTerminalModel(OpenAICompatModel):
    def __init__(self):
        super().__init__(MODEL,max_new_tokens=512)

    def complete_final(self,messages:list[dict],tools:list[dict])->AssistantTurn:
        if not messages or messages[0].get("role")!="system":
            raise ModelRuntimeError("model_error","Invalid terminal conversation")
        payload={
            "model":self.model,"messages":messages,
            "max_tokens":self.max_new_tokens,"temperature":0,
            "response_format":FINAL_SCHEMA,
        }
        # Exactly match the original OpenAICompatModel.complete_final payload
        # apart from adding response_format. Original final uses
        # tool_choice="none" WITH the original tools definitions present.
        if tools:
            payload["tools"]=tools
            payload["tool_choice"]="none"
        try:
            response=requests.post(f"{self.base_url}/v1/chat/completions",
                                   json=payload,timeout=self.timeout_s)
            response.raise_for_status()
            value=response.json()
        except requests.Timeout as exc:
            raise ModelRuntimeError("model_timeout",f"Constrained terminal timeout: {exc}") from None
        except requests.RequestException as exc:
            raise ModelRuntimeError("model_error",f"Constrained terminal HTTP failure: {exc}") from None
        except ValueError as exc:
            raise ModelRuntimeError("model_error",f"Constrained terminal response JSON invalid: {exc}") from None
        try:
            msg=value["choices"][0]["message"]
            if msg.get("tool_calls"):
                raise ModelRuntimeError("model_error","JSON final returned structured tool calls")
            raw=msg["content"]
            if not isinstance(raw,str):
                raise ModelRuntimeError("model_error","JSON final content is not text")
            answer=strictly_decode_answer(raw)
        except (KeyError,TypeError,IndexError) as exc:
            raise ModelRuntimeError("model_error",f"Constrained final protocol response malformed: {type(exc).__name__}") from exc
        return AssistantTurn(content=answer,tool_calls=[])


def select_frozen(dataset)->list[dict]:
    source=select_gaia_partition(dataset,partition="evaluation",seed=LOCAL_PROTOCOL_SEED)
    if len(source)!=100 or len({r["task_id"] for r in source})!=100:
        raise ValueError("GAIA100 original source changed")
    by_hash={sha256(str(r["task_id"]).encode()).hexdigest():r for r in source}
    if len(by_hash)!=100 or len(set(FROZEN_IDS))!=N or set(FROZEN_IDS)-set(by_hash):
        raise ValueError("Eight predeclared original GAIA DEV task identities missing")
    return [by_hash[task_id] for task_id in FROZEN_IDS]


def evaluate(rows:list[dict],arm:str)->list[dict]:
    if arm not in ARMS:raise ValueError("Unknown arm")
    old=runner.select_gaia_partition
    with TemporaryDirectory(prefix="gaia-structured-final-") as temp:
        root=Path(temp)
        try:
            runner.select_gaia_partition=lambda *_a,**_kw:rows
            runner.run_gaia100(
                rows,
                lambda:AgentRuntime(
                    ConstrainedTerminalModel() if arm=="json_schema_final"
                    else OpenAICompatModel(MODEL,max_new_tokens=512),
                    default_tools(),max_steps=12
                ),
                root,seed=LOCAL_PROTOCOL_SEED,partition="evaluation",
                dataset_revision=GAIA_REVISION
            )
        finally:
            runner.select_gaia_partition=old
        records=[json.loads(s) for s in (root/"results.jsonl").read_text().splitlines() if s.strip()]
    if len(records)!=len(rows):raise RuntimeError("Incomplete source-paired evaluation")
    return [{
        "task_hash":sha256(r["task_id"].encode()).hexdigest(),
        "correct":bool(r["correct"]),"completed":bool(r["completed"]),
        "stop_reason":r["stop_reason"],"tool_calls":int(r["tool_calls"]),
        "tool_errors":int(r["tool_errors"]),
        "duplicate_calls":int(r["duplicate_calls_blocked"]),
        "steps":int(r["steps"]),"latency_ms":round(float(r["latency_ms"]),2),
    } for r in records]


def run_shard(dataset,shard:int)->dict:
    if not 0<=shard<SHARD_COUNT:raise ValueError("Bad shard")
    fixed=select_frozen(dataset)
    # ABBA over case pairs to avoid making one arm always run first.
    cases=fixed[shard*2:shard*2+2]
    order=ARMS if shard%2==0 else tuple(reversed(ARMS))
    reports={a:evaluate(cases,a) for a in order}
    if [r["task_hash"] for r in reports["original"]] != [r["task_hash"] for r in reports["json_schema_final"]]:
        raise RuntimeError("Paired case identity drift")
    return {
        "status":"complete","shard":shard,"n":2,
        "source_revision":GAIA_REVISION,"selection":"already-seen-GAIA-dev-conditioned-smoke",
        "frozen_sha256":FROZEN_SHA,
        "model":"same-pinned-Qwen3.5-9B-v7-adapter",
        "tool_budget":12,"only_change":"final-step JSON Schema constrained decoding",
        "arms":reports,
    }


def aggregate(paths:list[Path])->dict:
    if len(paths)!=SHARD_COUNT:raise ValueError("Require all four physical shards")
    data=[json.loads(p.read_text()) for p in paths]
    if sorted(d.get("shard") for d in data)!=list(range(SHARD_COUNT)):
        raise ValueError("Duplicate/missing physical shard ID")
    if any(d.get("status")!="complete" or d.get("n")!=2 or
           d.get("source_revision")!=GAIA_REVISION or d.get("frozen_sha256")!=FROZEN_SHA or
           d.get("selection")!="already-seen-GAIA-dev-conditioned-smoke" or
           d.get("tool_budget")!=12 or set(d.get("arms",{}))!=set(ARMS) or
           any(len(d["arms"][a])!=2 for a in ARMS) for d in data):
        raise ValueError("Cohort/arm config drift")
    by_arm={arm:[r for d in sorted(data,key=lambda q:q["shard"]) for r in d["arms"][arm]] for arm in ARMS}
    if [r["task_hash"] for r in by_arm["original"]]!=list(FROZEN_IDS) or [r["task_hash"] for r in by_arm["json_schema_final"]]!=list(FROZEN_IDS):
        raise ValueError("Exactly eight frozen original task IDs required")
    a,b=by_arm["original"],by_arm["json_schema_final"]
    return {
        "status":"EXPOSED_DEV_MECHANISM_SCREEN_COMPLETE","n":N,
        "frozen_ids_sha256":FROZEN_SHA,
        "population":"4 prior invalid-final + 4 prior correct GAIA validation. Repeated, exposed, outcome-conditioned NOT a fresh unbiased benchmark.",
        "arms":{name:{
            "correct":sum(z["correct"] for z in values),
            "completed":sum(z["completed"] for z in values),
            "invalid_final":sum(z["stop_reason"]=="invalid_final" for z in values),
            "model_timeouts":sum(z["stop_reason"]=="model_timeout" for z in values),
            "model_errors":sum(z["stop_reason"]=="model_error" for z in values),
            "tool_calls":sum(z["tool_calls"] for z in values),
            "stop_reasons":dict(Counter(z["stop_reason"] for z in values)),
        } for name,values in by_arm.items()},
        "rescues":sum(not x["correct"] and y["correct"] for x,y in zip(a,b)),
        "regressions":sum(x["correct"] and not y["correct"] for x,y in zip(a,b)),
        "claim_boundary":"Mechanism-only small seen-GAIA diagnostic; no hidden test, no generalization or GAIA100 improvement. No promotion based on this screen alone.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.inputs:
        result=aggregate(args.inputs)
        print("GAIA_JSON_TERMINAL8_FINAL",json.dumps(result,sort_keys=True),flush=True)
    else:
        if args.shard is None:p.error("Need shard or inputs")
        result=run_shard(load_validation(),args.shard)
        print("GAIA_JSON_TERMINAL8_SHARD",args.shard,"complete",flush=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
