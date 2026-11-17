"""Prospective *diagnostic on already EXPOSED* public GAIA residual15.

Hypothesis: original v7 GGUF Qwen3.5-9B has a final-mode surface failure:
tool_choice='none' still yields a textual <tool_call> final. Replace ONLY a
failed final request (invalid textual tool, native tool call, empty response
or runtime timeout) with ONE constrained JSON-schema terminal call using
the SAME original live model, same history and tools and no new observations.
If the legacy final is valid, preserve it EXACTLY, ensuring zero regressions
on valid original finals by construction. Never train, change adapter, invoke
a stronger model, run a new search, or supply GAIA reference to the model.

CRITICAL: all 15 GAIA identities were previously evaluated (run 43467405441).
This is a diagnostic mechanism check, not unseen GAIA or hidden generalization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import requests

from gaia_small_agent.agent.loop import AgentRuntime,_looks_like_textual_tool_call
from gaia_small_agent.agent.types import AgentResult,AssistantTurn,ModelRuntimeError
from gaia_small_agent.benchmark import runner
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION,LOCAL_PROTOCOL_SEED,load_validation
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools
from scripts.eval_gaia_residual15_serializer import selected_residual,freeze as old_freeze,BASE_SHA,V7_SHA

PROTOCOL="GAIA-EXPOSED-RESIDUAL15-V7-LAST-FAILED-FINAL-JSON-GRAMMAR-V1"
N=15
SHARDS=3
MODEL="qwen3.5-9b"
SOURCE_SHA="aff003320eb310d0a8ce74f521432a26b76ba2adfa2d7f9d7f8ee244a7e8356c"
GATE={"net_rescues":2,"max_regressions":0,"min_levels":2,"min_completion_gain":2}
JSON_SCHEMA={"type":"object","properties":{"answer":{"type":"string"}},
             "required":["answer"],"additionalProperties":False}
GRAMMAR_SHA=sha256(json.dumps(JSON_SCHEMA,sort_keys=True).encode()).hexdigest()

def freeze(rows:list[dict])->dict:
    prior=old_freeze(rows)
    if prior["task_id_sequence_sha256"]!=SOURCE_SHA or prior["n"]!=N:
        raise RuntimeError("Exact previously used GAIA15 source changed")
    return {
        "status":"EXPOSED_HISTORICAL_GAIA15_SOURCE_REUSED_DIAGNOSTIC_ONLY",
        "protocol":PROTOCOL,"n":N,"shards":SHARDS,
        "source_revision":GAIA_REVISION,
        "task_id_sequence_sha256":SOURCE_SHA,
        "hashed_tasks":prior["hashed_tasks"],
        "source_seen_before":"GAIA terminal run 43467405441, 15/15 identical identities",
        "original_base_sha":BASE_SHA,"v7_adapter_sha":V7_SHA,
        "structured_grammar_sha256":GRAMMAR_SHA,
        "no_gaia_gold_in_inference":True,
        "same_unmodified_original_agent_until_final":True,
        "fallback_only_after_failed_terminal":True,
        "no_new_tool_execution_on_fallback":True,
        "preregistered_gate":GATE,
        "cannot_promote_or_claim_hidden_generalization":True,
    }

def _turn_is_valid(turn:AssistantTurn)->bool:
    answer=(turn.content or "").strip()
    return bool(not turn.tool_calls and answer
                and not _looks_like_textual_tool_call(answer))

def _request_json_final(model:OpenAICompatModel,messages:list[dict],tools:list[dict])->tuple[str,str]:
    """Pinned llama-server documented response_format=json_object + schema.

    Parse and validate independently; do not trust grammar enforcement alone.
    """
    payload={
        "model":model.model,"messages":messages,
        "max_tokens":model.max_new_tokens,
        "temperature":0,
        "response_format":{"type":"json_object","schema":JSON_SCHEMA},
    }
    if tools:
        payload["tools"]=tools
        payload["tool_choice"]="none"
    try:
        resp=requests.post(model.base_url+"/v1/chat/completions",
                           json=payload,timeout=model.timeout_s)
        resp.raise_for_status()
        data=resp.json()
        choices=data["choices"]
        message=choices[0]["message"]
        if message.get("tool_calls"):
            return "","structured_tool_call"
        raw=message.get("content")
        if not isinstance(raw,str):
            return "","structured_empty"
        obj=json.loads(raw)
        if not isinstance(obj,dict) or set(obj)!={"answer"} or not isinstance(obj["answer"],str):
            return "","schema_not_honored"
        ans=obj["answer"].strip()
        if (not ans or len(ans)>512 or _looks_like_textual_tool_call(ans)
            or ans.startswith("{") and '"name"' in ans and '"arguments"' in ans):
            return "","invalid_answer"
        return ans,"accepted_schema"
    except (requests.Timeout,requests.ConnectionError):
        return "","structured_timeout"
    except (requests.RequestException,KeyError,IndexError,TypeError,ValueError,
            json.JSONDecodeError):
        return "","structured_error"

def server_schema_preflight(base_url:str="http://127.0.0.1:8080")->bool:
    """Source-free probe, fail-closed if exact schema isn't produced by pinned llama."""
    m=OpenAICompatModel(model=MODEL,base_url=base_url,timeout_s=300,max_new_tokens=96)
    answer,status=_request_json_final(m,[
        {"role":"system","content":"Respond to the test request with the word ready in the answer field only."},
        {"role":"user","content":"Output exactly ready."},
    ],[])
    return status=="accepted_schema" and answer.casefold()=="ready"

class FailedFinalGrammarModel(OpenAICompatModel):
    """Record identical original terminal, only attempt grammar if invalid."""
    def __init__(self):
        super().__init__(model=MODEL,max_new_tokens=512)
        self.terminals=[]

    def complete_final(self,messages:list[dict],tools:list[dict])->AssistantTurn:
        start=perf_counter()
        original=None
        error=None
        try:
            original=super().complete_final(messages,tools)
        except ModelRuntimeError as exc:
            error=exc
        original_elapsed=perf_counter()-start
        if original is not None and _turn_is_valid(original):
            self.terminals.append({
                "original_completed":True,"original_answer":original.content.strip(),
                "original_status":"valid_final","fallback_attempted":False,
                "fallback_status":"not_needed",
                "original_request_s":round(original_elapsed,2),"fallback_request_s":0.0,
            })
            return original
        original_status=(error.stop_reason if error is not None else
                         "tool_call" if original is not None and original.tool_calls else
                         "invalid_final" if original is not None and
                         _looks_like_textual_tool_call(original.content or "") else
                         "empty_final")
        second=perf_counter()
        answer,status=_request_json_final(self,messages,tools)
        second_elapsed=perf_counter()-second
        self.terminals.append({
            "original_completed":False,"original_answer":"",
            "original_status":original_status,"fallback_attempted":True,
            "fallback_status":status,
            "original_request_s":round(original_elapsed,2),
            "fallback_request_s":round(second_elapsed,2),
        })
        if status=="accepted_schema":
            return AssistantTurn(content=answer,tool_calls=[])
        if error is not None:
            raise error
        if original is None:
            raise ModelRuntimeError("model_error","Original and grammar final failed")
        return original

class PairedRuntime:
    def __init__(self):
        self.model=FailedFinalGrammarModel()
        self.agent=AgentRuntime(self.model,default_tools(),max_steps=12)
        self.case_snapshots=[]

    def run(self,question:str,workspace:Path)->AgentResult:
        old_n=len(self.model.terminals)
        got=self.agent.run(question,workspace)
        terminal=self.model.terminals[old_n:]
        if len(terminal)>1:
            raise RuntimeError("More than one final request in one GAIA case")
        t=terminal[0] if terminal else None
        # Any difference from original can only happen at a failed final request.
        original_completed=(got.completed if t is None else t["original_completed"])
        original_answer=(got.answer if t is None else t["original_answer"])
        if t is None and got.metrics.steps==12:
            raise RuntimeError("Final step missing terminal capture")
        self.case_snapshots.append({
            "original_completed":original_completed,
            "original_answer":original_answer,
            "original_status":(got.stop_reason if t is None else t["original_status"]),
            "fallback_attempted":bool(t and t["fallback_attempted"]),
            "fallback_status":("not_needed" if t is None else t["fallback_status"]),
            "fallback_s":0 if t is None else t["fallback_request_s"],
            "tool_calls":got.metrics.tool_calls,"tool_errors":got.metrics.tool_errors,
        })
        return got

def run_shard(rows:list[dict],locked:dict,shard:int)->dict:
    if not 0<=shard<SHARDS or freeze(rows)!=locked:
        raise RuntimeError("Previously exposed original GAIA source/grammar lock drift")
    selected=selected_residual(rows)
    subset=shard_selected_rows(selected,shard_index=shard,shard_count=SHARDS)
    if len(subset)!=5:
        raise RuntimeError("Residual source does not have 5 rows")
    if not server_schema_preflight():
        raise RuntimeError("Pinned physical llama-server does not enforce and return JSON")
    rt=PairedRuntime()
    original_selector=runner.select_gaia_partition
    try:
        with TemporaryDirectory(prefix="gaia-v7-final-grammar-") as tmp:
            runner.select_gaia_partition=lambda *args,**kwargs:subset
            runner.run_gaia100(subset,lambda:rt,Path(tmp),seed=LOCAL_PROTOCOL_SEED,
                               dataset_revision=GAIA_REVISION)
            results=[json.loads(z) for z in (Path(tmp)/"results.jsonl").read_text().splitlines()]
    finally:
        runner.select_gaia_partition=original_selector
    if len(results)!=5 or len(rt.case_snapshots)!=5:
        raise RuntimeError("Incomplete physically paired 5-task trajectories")
    pairs=[]
    for source,graded,legacy in zip(subset,results,rt.case_snapshots):
        if source["task_id"]!=graded["task_id"]:
            raise RuntimeError("GAIA physical scorer/source ordering drift")
        orig=bool(legacy["original_completed"] and
                  gaia_score(legacy["original_answer"],str(source["Final answer"])))
        final=bool(graded["correct"])
        if orig and not final:
            # Should be impossible; original final text is unchanged if valid.
            raise RuntimeError("Baseline-correct final unexpectedly regressed")
        if graded["tool_calls"]!=legacy["tool_calls"]:
            raise RuntimeError("Tool trace mutated by grammar")
        pairs.append({
            "task_hash":sha256(source["task_id"].encode()).hexdigest(),
            "level":int(source["Level"]),"original_correct":orig,
            "grammar_correct":final,"original_completed":legacy["original_completed"],
            "grammar_completed":bool(graded["completed"]),
            "original_status":legacy["original_status"],
            "grammar_status":str(graded["stop_reason"]),
            "fallback_attempted":legacy["fallback_attempted"],
            "fallback_status":legacy["fallback_status"],
            "fallback_s":legacy["fallback_s"],
            "tool_calls":legacy["tool_calls"],"tool_errors":legacy["tool_errors"],
            "same_tools_and_history":True,
        })
    return {
        "status":"complete","protocol":PROTOCOL,
        "source_sha":SOURCE_SHA,"grammar_sha":GRAMMAR_SHA,
        "base_sha":BASE_SHA,"adapter_sha":V7_SHA,
        "shard":shard,"n":5,"cases":pairs,
    }

def aggregate(shards:list[dict])->dict:
    if (len(shards)!=SHARDS or sorted(s.get("shard") for s in shards)!=list(range(SHARDS))
        or any(s.get("status")!="complete" or s.get("protocol")!=PROTOCOL
               or s.get("source_sha")!=SOURCE_SHA
               or s.get("grammar_sha")!=GRAMMAR_SHA
               or s.get("base_sha")!=BASE_SHA or s.get("adapter_sha")!=V7_SHA
               or s.get("n")!=5 or len(s.get("cases",[]))!=5 for s in shards)):
        raise RuntimeError("Not all physical same-v7 grammar shards completed")
    pairs=[c for s in sorted(shards,key=lambda x:x["shard"]) for c in s["cases"]]
    if len(pairs)!=N or len({c["task_hash"] for c in pairs})!=N:
        raise RuntimeError("Incomplete/duplicate GAIA source identities")
    rescue=[x for x in pairs if x["grammar_correct"] and not x["original_correct"]]
    regress=[x for x in pairs if x["original_correct"] and not x["grammar_correct"]]
    original=sum(x["original_correct"] for x in pairs)
    changed=sum(x["grammar_correct"] for x in pairs)
    old_completed=sum(x["original_completed"] for x in pairs)
    now_completed=sum(x["grammar_completed"] for x in pairs)
    passed=(len(rescue)-len(regress)>=GATE["net_rescues"]
            and len(regress)<=GATE["max_regressions"]
            and len({x["level"] for x in rescue})>=GATE["min_levels"]
            and now_completed-old_completed>=GATE["min_completion_gain"])
    return {
        "status":"COMPLETE_EXPOSED_GAIA15_JSON_GRAMMAR_DIAGNOSTIC",
        "protocol":PROTOCOL,"source_sha":SOURCE_SHA,
        "base_sha":BASE_SHA,"adapter_sha":V7_SHA,
        "structured_grammar_sha":GRAMMAR_SHA,
        "n":N,"original_correct":original,"grammar_correct":changed,
        "original_completed":old_completed,"grammar_completed":now_completed,
        "rescues":len(rescue),"regressions":len(regress),
        "net_exact":changed-original,
        "rescues_by_level":dict(sorted(Counter(x["level"] for x in rescue).items())),
        "legacy_stops":dict(Counter(x["original_status"] for x in pairs)),
        "fallback_states":dict(Counter(x["fallback_status"] for x in pairs)),
        "extra_terminal_calls":sum(x["fallback_attempted"] for x in pairs),
        "extra_terminal_seconds":round(sum(x["fallback_s"] for x in pairs),2),
        "tool_calls_unchanged":sum(x["tool_calls"] for x in pairs),
        "tool_errors_unchanged":sum(x["tool_errors"] for x in pairs),
        "predeclared_gate":GATE,"positive_diagnostic_only":passed,
        "promotion_allowed":False,
        "EXPOSED_prior_GAIA15":True,
        "claim_boundary":"All 15 public GAIA validation tasks are historically exposed (Actions 43467405441); strict same-run paired source-independent terminal fallback only, old valid final preserved verbatim, additional same-v7 JSON grammar final call on failed final, no extra tool actions/gold or new weights. A success does NOT demonstrate untouched, hidden or official GAIA improvement and cannot be promoted without independent held-out transfer. Model capacity/timeouts, no relevant evidence and absent multimodal support are not solved.",
        "cases":pairs,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only",action="store_true")
    p.add_argument("--probe-server",action="store_true")
    p.add_argument("--frozen",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.probe_server:
        if not server_schema_preflight():
            raise RuntimeError("JSON schema final preflight failed on physical model")
        print("GAIA_JSON_GRAMMAR_PHYSICAL_PREFLIGHT_SUCCESS",GRAMMAR_SHA,flush=True)
        return
    if a.output is None:
        p.error("--output required")
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.inputs:
        out=aggregate([json.loads(path.read_text()) for path in a.inputs])
        print("GAIA_GRAMMAR15_FINAL",json.dumps({k:v for k,v in out.items() if k!="cases"},
            sort_keys=True),flush=True)
    else:
        rows=load_validation()
        out=freeze(rows) if a.source_only else run_shard(rows,json.loads(a.frozen.read_text()),a.shard)
        if a.source_only:
            print("GAIA_GRAMMAR15_EXPOSED_SOURCE_LOCK",SOURCE_SHA,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")

if __name__=="__main__":
    main()
