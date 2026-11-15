"""Physical GPU evaluation of Qwen9B original vs its real 8-step NextSearch LoRA.

Validation questions come ONLY from the earlier immutable NextSearch VAL
freeze (#43722732105); no training task or GAIA answer enters the model.
Same loaded 4bit Qwen3.5-9B, same native original AgentRuntime and live
source-compatible search/fetch; use PeftModel.disable_adapter for matched
unadapted control, never a different Ollama/transformers comparison.

Small 4-case ABBA screen, not an independent hidden scientific gain/GAIA score.
Both arms use the identical prompt and tools; website access can vary
between sequential runs, so report tool success and control order too.
"""
from __future__ import annotations
from collections import Counter
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any

COHORT_SHA="4cf774e6bf76210ff33cf9c41ea3374728aadf12d2d20510224b56fc116a73cd"
ADAPTER_SHA="69deffc3ca0548778cf451332b76ee19073780d0490baba22193a36ed19c3508"
SOURCE_SHA="9f2af9769d24b702730774c21a9fe057b0f29131"
MODEL_REV="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL="Qwen/Qwen3.5-9B"
CODE_SHA="__CODE_SHA__"
N=4
SOURCE_IDS=2
MAX_STEPS=9
MAX_NEW_TOKENS=256
WORK=Path("/content/nextsearch-abba-real-8step-val4")
INPUT=Path("/content/nextsearch-pinned-val8.json")
ADAPTER_DIR=Path("/content/nextsearch-original-8step-adapter")
FINAL=Path("/content/nextsearch-qwen9b-adapter-vs-base-val4.json")
PROTOCOL="nextsearch-qwen35-9b-original8step-val4-matched-transformers4bit-abba-v1"


def json_dump(x):
    return json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2,default=str)


def frozen_contract()->list[dict]:
    if len(CODE_SHA)!=40 or CODE_SHA=="__CODE_SHA__":
        raise ValueError("source code revision must be pinned")
    cohort=json.loads(INPUT.read_text())
    cases=cohort.get("cases") if isinstance(cohort,dict) else None
    if (cohort.get("cases_sha")!=COHORT_SHA
        or cohort.get("split")!="validation"
        or cohort.get("heldout_by_task_id") is not True
        or cohort.get("no_gold_to_model") is not True
        or not isinstance(cases,list) or len(cases)!=8
        or sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()!=COHORT_SHA):
        raise ValueError("source-disjoint VAL8 original manifest drift")
    selected=cases[:N]
    if (len(selected)!=N or len({r["id"] for r in selected})!=N
        or len({r["source"] for r in selected})<SOURCE_IDS):
        raise ValueError("four source-diverse heldout identities unavailable")
    if sha256((ADAPTER_DIR/"adapter_model.safetensors").read_bytes()).hexdigest()!=ADAPTER_SHA:
        raise ValueError("real checkpoint adapter SHA256 mismatch")
    conf=json.loads((ADAPTER_DIR/"adapter_config.json").read_text())
    targets=set(conf.get("target_modules") or [])
    if targets!={"q_proj","v_proj"}:
        raise ValueError("real LoRA module target set changed")
    return selected


def status(res,elapsed:float)->dict[str,Any]:
    from gaia_small_agent.benchmark.scoring import gaia_score
    names=Counter(e.data.get("name") for e in res.trace if e.kind=="tool_call")
    tool_results=[e for e in res.trace if e.kind=="tool_result"]
    return {
        "answer":res.answer,
        "completed":res.completed,
        "stop_reason":res.stop_reason,
        "steps":res.metrics.steps,
        "tool_calls":res.metrics.tool_calls,
        "tool_errors":res.metrics.tool_errors,
        "invalid_tool_calls":res.metrics.invalid_tool_calls,
        "duplicate_calls_blocked":res.metrics.duplicate_calls_blocked,
        "tool_result_success":sum(e.data.get("ok") is True for e in tool_results),
        "tool_result_error":sum(e.data.get("ok") is False for e in tool_results),
        "tool_names":dict(names),
        "duration_s":round(elapsed,2),
    }


def judge(res,reference:str)->bool:
    from gaia_small_agent.benchmark.scoring import gaia_score
    return bool(res["completed"] and gaia_score(res["answer"],reference))


def initial_prompt(question:str)->str:
    return (
        "Answer the question using public web research. The search tool accepts "
        "an objective and search_queries; the fetch tool reads public URLs. "
        "Check relevant sources and return only the final short answer. "
        "Do not cite an answer you have not verified.\n\nQuestion: "+question
    )


def one(model,case:dict,arm:str,workspace:Path)->dict:
    from gaia_small_agent.agent.loop import AgentRuntime
    from scripts.nextsearch_source_tools import ResearchSearch,ResearchFetch
    if arm not in ("original","trained8"):
        raise ValueError("invalid original vs adapter evaluation arm")
    ctx=model.model.disable_adapter() if arm=="original" else nullcontext()
    started=monotonic()
    with ctx:
        answer=AgentRuntime(model,[ResearchSearch(),ResearchFetch()],
                            max_steps=MAX_STEPS).run(
                                initial_prompt(case["question"]),workspace
                            )
    rec=status(answer,monotonic()-started)
    rec["arm"]=arm
    return rec


def aggregate(rows:list[dict])->dict:
    if len(rows)!=N or len({r["id"] for r in rows})!=N:
        raise ValueError("missing heldout tasks")
    if any(set(r.get("arms",{}))!={"original","trained8"} for r in rows):
        raise ValueError("missing physical GPU pair")
    before=sum(r["arms"]["original"]["strict_exact"] for r in rows)
    after=sum(r["arms"]["trained8"]["strict_exact"] for r in rows)
    rescues=[r for r in rows if not r["arms"]["original"]["strict_exact"]
             and r["arms"]["trained8"]["strict_exact"]]
    regress=[r for r in rows if r["arms"]["original"]["strict_exact"]
             and not r["arms"]["trained8"]["strict_exact"]]
    sum_tool=lambda name:sum(r["arms"][name]["tool_calls"] for r in rows)
    sum_err=lambda name:sum(r["arms"][name]["tool_errors"] for r in rows)
    all_good=all(r["arms"][arm]["stop_reason"] not in ("model_error","model_capacity","model_timeout","model_unavailable")
                 for r in rows for arm in ("original","trained8"))
    return {
        "protocol":PROTOCOL,"source_cohort_sha":COHORT_SHA,
        "adapter_sha":ADAPTER_SHA,
        "base_model_revision":MODEL_REV,
        "cases":N,"source_count":len({r["source"] for r in rows}),
        "base_strict_exact":before,
        "adapter_strict_exact":after,
        "rescue_ids":[r["id"] for r in rescues],
        "regression_ids":[r["id"] for r in regress],
        "baseline_tool_calls":sum_tool("original"),
        "adapter_tool_calls":sum_tool("trained8"),
        "baseline_tool_errors":sum_err("original"),
        "adapter_tool_errors":sum_err("trained8"),
        "all_model_calls_healthy":all_good,
        "pass_small_screen_only":bool(all_good and len(rescues)>=2 and not regress
                                     and len({r["source"] for r in rescues})>=2
                                     and sum_tool("trained8")<=sum_tool("original")+2
                                     and sum_err("trained8")<=sum_err("original")),
        "promoted":False,"gaia100_tested":False,
        "public_source_web_live_order_caveat":True,
        "reported_main_gaia100_unchanged":18,
        "rows":rows,
    }


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No Colab CUDA GPU; cannot call this an actual adapter test")
    props=torch.cuda.get_device_properties(0)
    if props.total_memory/2**30<14.5:
        raise RuntimeError("T4-class minimum memory not met")
    cases=frozen_contract()
    if WORK.exists() or FINAL.exists():
        raise FileExistsError("cannot overwrite any prior heldout test files")
    command=[
        "git","clone","--branch","experiment/nextsearch-gpu8-heldout-abba-val4-20261116",
        "--depth","10","https://github.com/kolihhan/small-agent-qlora.git","/content/agent-eval-source"
    ]
    subprocess.run(command,check=True)
    subprocess.run(["git","-C","/content/agent-eval-source","checkout","--detach",CODE_SHA],check=True)
    checkout=subprocess.check_output(["git","-C","/content/agent-eval-source","rev-parse","HEAD"],text=True).strip()
    if checkout!=CODE_SHA:
        raise ValueError("audited code revision changed")
    subprocess.run([sys.executable,"-m","pip","install","-q","-e","/content/agent-eval-source[train,search]"],
                   check=True)
    sys.path.insert(0,"/content/agent-eval-source/src")
    sys.path.insert(0,"/content/agent-eval-source")
    from huggingface_hub import snapshot_download
    from gaia_small_agent.model.transformers_qwen import TransformersQwenModel
    resolved=snapshot_download(MODEL,revision=MODEL_REV)
    model=TransformersQwenModel(
        model_name=resolved,quantize_4bit=True,adapter=str(ADAPTER_DIR),
        enable_thinking=False,max_new_tokens=MAX_NEW_TOKENS,
    )
    import peft
    if not isinstance(model.model,peft.PeftModel):
        raise RuntimeError("Real PEFT adapter was not loaded on GPU")
    if not callable(getattr(model.model,"disable_adapter",None)):
        raise RuntimeError("Cannot isolate base model from adapter in identical runtime")
    print("NEXTSEARCH_PAIR_T4_SAME_MODEL_LOADED",
          json.dumps({"gpu":props.name,"vram_gib":round(props.total_memory/2**30,2),
          "adapter_sha256":ADAPTER_SHA,"source_cohort_sha":COHORT_SHA,
          "max_new_tokens":MAX_NEW_TOKENS,"max_steps":MAX_STEPS,
          "no_training":True,"training_source_split":"train",
          "eval_source_split":"validation"},sort_keys=True),flush=True)
    records=[]
    for index,case in enumerate(cases):
        # ABBA counterbalance for live-web sequence effects.
        order=("original","trained8") if index in (0,3) else ("trained8","original")
        rec={"id":case["id"],"source":case["source"],"order":list(order),"arms":{}}
        for arm in order:
            result=one(model,case,arm,WORK/f"{index}-{arm}")
            # Reference is opened only after this model episode completes.
            result["strict_exact"]=judge(result,case["answer"])
            rec["arms"][arm]=result
            print("NEXTSEARCH_8STEP_VAL4_PAIRED_REAL",
                  index,case["source"],arm,"exact",result["strict_exact"],
                  "tool_calls",result["tool_calls"],
                  "stop",result["stop_reason"],flush=True)
        records.append(rec)
        WORK.mkdir(parents=True,exist_ok=True)
        (WORK/"interim.json").write_text(json_dump({
            "protocol":PROTOCOL,"completed_case_count":len(records),
            "all_rows":records
        })+"\n")
    report=aggregate(records)
    FINAL.write_text(json_dump(report)+"\n")
    print("NEXTSEARCH_QWEN9B_REAL_ADAPTER_VS_BASE_VAL4",
          json.dumps({k:v for k,v in report.items() if k!="rows"},
                     ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
