"""Real source-disjoint 2Wiki local-doc heldout Qwen9B base-vs-trained8.

Offline local articles replace the broken public search service. Both arms
share one original 4-bit Qwen3.5-9B model and the original AgentRuntime,
identical SEARCH and READ schemas, same prompt, same fixed 2Wiki validation
publisher context, and PEFT.disable_adapter() for strict within-model control.
No source answers/support spans enter the model's prompt or tool results.
No new adapter training. Source VAL task IDs and source files were frozen in
physical independent 2Wiki validation action run #43712873154.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic

PROTOCOL="qwen35-9b-trained8-original-local-2wiki-validation4-paired-abba-v1"
CODE_SHA="__CODE_SHA__"
MODEL="Qwen/Qwen3.5-9B"
MODEL_REV="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADAPTER_SHA="69deffc3ca0548778cf451332b76ee19073780d0490baba22193a36ed19c3508"
COHORT_SHA="fded3c65e2ccc829ae0efdd8af448896320590b63bb69e0e9ff1d52eadc4fbb7"
N=4
MAX_STEPS=10
MAX_NEW_TOKENS=384
COHORT=Path("/content/2wiki-original-val8-source.json")
ADAPTER=Path("/content/qwen9b-trained8-adapter")
WORK=Path("/content/qwen9b-2wiki-local-pair-work")
FINAL=Path("/content/qwen9b-original8-2wiki-local-val4-pair.json")


def validate_source():
    if CODE_SHA=="__CODE_SHA__" or len(CODE_SHA)!=40:
        raise RuntimeError("exact evaluation script source revision not pinned")
    manifest=json.loads(COHORT.read_text())
    cases=manifest["cases"]
    if (manifest["protocol"]!="2wiki-validation-qwen9b-visible-local-links-pair-v1"
        or manifest["source_revision"]!="fe713bfbd1afbca1a65246741a75890405d56a3a"
        or manifest["split"]!="validation"
        or len(cases)!=8
        or manifest["cases_sha"]!=COHORT_SHA
        or sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()!=COHORT_SHA):
        raise RuntimeError("original frozen VAL8 source changed")
    selected=cases[:N]
    if len({c["id"] for c in selected})!=N:
        raise RuntimeError("missing unique source VAL identities")
    for c in selected:
        if not (len(c["files"])>=6 and len(c["support"])==4 and
                set(c["support"])<=set(c["files"]) and
                set(c["index_titles"].values())==set(c["files"])):
            raise RuntimeError("source-grounded local pages missing or invalid")
    if sha256((ADAPTER/"adapter_model.safetensors").read_bytes()).hexdigest()!=ADAPTER_SHA:
        raise RuntimeError("physical original trained adapter SHA drift")
    peft=json.loads((ADAPTER/"adapter_config.json").read_text())
    if set(peft.get("target_modules") or [])!={"q_proj","v_proj"}:
        raise RuntimeError("trained q/v adapters not same original")
    return selected


def local_components():
    from gaia_small_agent.tools.base import Tool,ToolResult
    class LocalSearch(Tool):
        name="search"
        description="Search local article titles/snippets by a named entity or topic. Returns article file paths for the read tool. Try another related name if no results."
        schema={"type":"object","properties":{"query":{"type":"string","minLength":2}},
                "required":["query"],"additionalProperties":False}
        def __init__(self,title_paths:dict[str,str],files:dict[str,str]):
            self.paths,self.files=title_paths,files
        def run(self,arguments,workspace):
            query=str(arguments.get("query") or "").casefold().strip()
            tokens=set(re.findall(r"[a-z0-9]{3,}",query))-{
                "what","which","that","whose","when","where","year","name"
            }
            scored=[]
            for title,path in self.paths.items():
                name=title.casefold()
                overlap=sum(token in name for token in tokens)
                phrase=int(bool(query and (query in name or name in query)))
                score=4*phrase+overlap
                if score:
                    scored.append((score,title,path))
            scored.sort(key=lambda x:(-x[0],x[1].casefold()))
            results=[
                f"{title} | {path} | {self.files[path][:110]}"
                for _,title,path in scored[:5]
            ]
            return ToolResult(True,"\n".join(results) if results else "No matching local article found.")
    return LocalSearch


def observe(case,result):
    from gaia_small_agent.benchmark.scoring import gaia_score
    pending={}
    delivered=defaultdict(list)
    names=Counter()
    successes=0
    for event in result.trace:
        if event.kind=="tool_call":
            names[event.data["name"]]+=1
            if event.data["name"]=="read":
                path=event.data.get("arguments",{}).get("source")
                if isinstance(path,str):
                    pending[event.data["id"]]=path
        if event.kind=="tool_result" and event.data.get("ok"):
            successes+=1
            path=pending.get(event.data.get("id"))
            if path:
                text=event.data.get("content","")
                if text.startswith("ToolMeta: "):
                    text=text.partition("\n")[2]
                delivered[path].append(text)
    visible=0
    for path,spans in case["support"].items():
        if any(all(str(span).casefold() in text.casefold() for span in spans)
               for text in delivered[path]):
            visible+=1
    return {
        "answer":result.answer,
        "completed":result.completed,
        "stop_reason":result.stop_reason,
        "strict_exact":bool(result.completed and gaia_score(result.answer,case["answer"])),
        "visible_support_docs":visible,
        "full_annotated_support":visible==4,
        "steps":result.metrics.steps,
        "tool_calls":result.metrics.tool_calls,
        "tool_errors":result.metrics.tool_errors,
        "tool_successes":successes,
        "search_calls":names["search"],
        "read_calls":names["read"],
        "duplicate_blocks":result.metrics.duplicate_calls_blocked,
        "invalid_calls":result.metrics.invalid_tool_calls,
    }


def evaluate(model,case,arm,workspace):
    from gaia_small_agent.tools import ReadTool
    from gaia_small_agent.agent.loop import AgentRuntime
    from gaia_small_agent.training.trajectories import _materialize_files
    LocalSearch=local_components()
    _materialize_files({"files":case["files"]},workspace)
    search=LocalSearch(case["index_titles"],case["files"])
    read=ReadTool()
    # Mandatory genuine tool preflight before model: no oracle labels.
    trial=search.run({"query":next(iter(case["index_titles"]))},workspace)
    if not trial.ok or "pages/" not in trial.content:
        raise RuntimeError("actual local first-party source search did not return any path")
    read_path=case["index_titles"][next(iter(case["index_titles"]))]
    observed=read.run({"source":read_path,"max_chars":2000},workspace)
    if not observed.ok or not observed.content:
        raise RuntimeError("actual local read failed; cannot run meaningful QLoRA comparison")
    prompt=(
        "Answer the following question using the local article search and read tools. "
        "Search for article names and read returned paths before answering. "
        "A local source may be a distractor; decide whether more evidence is needed. "
        "Return only the answer, not reasoning.\n\nQuestion: "+case["question"]
    )
    ctx=model.model.disable_adapter() if arm=="original" else nullcontext()
    start=monotonic()
    with ctx:
        got=AgentRuntime(model,[search,read],max_steps=MAX_STEPS).run(prompt,workspace)
    info=observe(case,got)
    info["seconds"]=round(monotonic()-start,2)
    info["arm"]=arm
    return info


def report(rows):
    if len(rows)!=N or len({r["id"] for r in rows})!=N:
        raise RuntimeError("incomplete paired independent 2Wiki source cases")
    score=lambda arm:sum(r["arms"][arm]["strict_exact"] for r in rows)
    count=lambda arm,key:sum(r["arms"][arm][key] for r in rows)
    rescued=[r for r in rows if not r["arms"]["original"]["strict_exact"]
             and r["arms"]["trained8"]["strict_exact"]]
    regressed=[r for r in rows if r["arms"]["original"]["strict_exact"]
              and not r["arms"]["trained8"]["strict_exact"]]
    base,adapt=score("original"),score("trained8")
    tools_before,tools_after=count("original","tool_calls"),count("trained8","tool_calls")
    good_sources=all(r["arms"][a]["tool_successes"]>0 for r in rows
                     for a in ("original","trained8"))
    return {
        "protocol":PROTOCOL,
        "cohort_sha":COHORT_SHA,
        "adapter_sha":ADAPTER_SHA,
        "model_revision":MODEL_REV,
        "scorable":N,
        "original_strict_exact":base,
        "trained8_strict_exact":adapt,
        "rescue_ids":[r["id"] for r in rescued],
        "regression_ids":[r["id"] for r in regressed],
        "base_tool_calls":tools_before,
        "trained8_tool_calls":tools_after,
        "base_tool_errors":count("original","tool_errors"),
        "trained8_tool_errors":count("trained8","tool_errors"),
        "base_tool_successes":count("original","tool_successes"),
        "trained8_tool_successes":count("trained8","tool_successes"),
        "base_full_support":count("original","full_annotated_support"),
        "trained8_full_support":count("trained8","full_annotated_support"),
        "tool_environment_healthy":good_sources,
        "positive_screen_only":bool(
            good_sources and len(rescued)>=2 and not regressed and
            tools_after<=tools_before+2 and
            count("trained8","tool_errors")<=count("original","tool_errors")),
        "promotion_allowed":False,
        "gaia100_new_score_claim":False,
        "original_gaia100_remains":18,
        "source_note":"original pinned independent 2Wiki VALIDATION, public previously evaluated by native Qwen; no training overlap, not pristine hidden OOD",
        "rows":rows,
    }


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("genuine Colab GPU required; no CPU model substitution")
    if torch.cuda.get_device_properties(0).total_memory/2**30<14.5:
        raise RuntimeError("minimum 14.5 GiB GPU absent")
    if WORK.exists() or FINAL.exists() or ADAPTER.exists():
        raise FileExistsError("refuse to overwrite earlier physical evaluation")
    ADAPTER.mkdir(parents=True)
    for name in ("adapter_model.safetensors","adapter_config.json"):
        from_path=Path("/content")/name
        if not from_path.is_file():
            raise RuntimeError(f"staged PEFT original checkpoint missing: {name}")
        (ADAPTER/name).write_bytes(from_path.read_bytes())
    cases=validate_source()
    source=Path("/content/qwen9b-eval-source")
    subprocess.run(["git","clone","--branch","experiment/nextsearch-trained8-2wiki-local-val4-20261116",
                    "--depth","10","https://github.com/kolihhan/small-agent-qlora.git",str(source)],check=True)
    subprocess.run(["git","-C",str(source),"checkout","--detach",CODE_SHA],check=True)
    sha=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if sha!=CODE_SHA:
        raise RuntimeError("source checkout drift")
    subprocess.run([sys.executable,"-m","pip","install","-q","-e",str(source)+"[train]"],check=True)
    sys.path[:0]=[str(source/"src"),str(source)]
    from huggingface_hub import snapshot_download
    from gaia_small_agent.model.transformers_qwen import TransformersQwenModel
    from peft import PeftModel
    resolved=snapshot_download(MODEL,revision=MODEL_REV)
    model=TransformersQwenModel(
        model_name=resolved,quantize_4bit=True,adapter=str(ADAPTER),
        enable_thinking=False,max_new_tokens=MAX_NEW_TOKENS
    )
    if not isinstance(model.model,PeftModel):
        raise RuntimeError("no PEFT adapter loaded")
    print("NEXTSEARCH_2WIKI_SOURCE_LOCAL_GPU_READY",
          json.dumps({"gpu":torch.cuda.get_device_properties(0).name,
          "model":MODEL,"adapter_sha256":ADAPTER_SHA,
          "cohort_sha256":COHORT_SHA,"cases":N,
          "source_preflight":"real search+read required",
          "no_network_web_search":True,"no_finetuning":True},
          sort_keys=True),flush=True)
    rows=[]
    for idx,case in enumerate(cases):
        order=("original","trained8") if idx in (0,3) else ("trained8","original")
        out={"id":case["id"],"order":list(order),"arms":{}}
        for arm in order:
            res=evaluate(model,case,arm,WORK/f"{idx}-{arm}")
            out["arms"][arm]=res
            print("NEXTSEARCH_2WIKI_REAL_PAIR",idx,arm,"exact",res["strict_exact"],
                  "success",res["tool_successes"],
                  "search",res["search_calls"],"read",res["read_calls"],
                  "completed",res["completed"],flush=True)
        rows.append(out)
        WORK.mkdir(parents=True,exist_ok=True)
        (WORK/"interim.json").write_text(json.dumps({
            "protocol":PROTOCOL,"completed":len(rows),"rows":rows
        },indent=2,sort_keys=True)+"\n")
    final=report(rows)
    FINAL.write_text(json.dumps(final,sort_keys=True,indent=2)+"\n")
    print("NEXTSEARCH_TRAINED8_VS_BASE_LOCAL_2WIKI_VAL4_FINAL",
          json.dumps({k:v for k,v in final.items() if k!="rows"},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
