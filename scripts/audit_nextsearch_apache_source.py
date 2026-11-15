"""Original NextSearch Apache-2.0 web research agent SFT source quality audit.

Source-native OpenAI format, not our Qwen agent's tool contract.
Inspect bounded search/fetch observations, strict answer match, nontrivial
multi-tool paths, independent source IDs, and observed error recovery.
No raw prompts, thinking, web text, or reference answers are exported.
No GAIA benchmark content and no model training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

from gaia_small_agent.benchmark.scoring import gaia_score

REPO = "NextTokenAI/NextSearch-1-Trajectories"
CONFIG = "sft-trajectories"
PROTOCOL = "nextsearch-apache-original-web-agent-trajectory-readiness-v1"
MAX_SOURCE_ROWS = 3270
FROZEN_REVISION = "d32183c6289d32c5d51b6247e68d53a58dddd495"
MIN_OBSERVED = 7


def _call(call):
    if not isinstance(call, dict):
        return None, None, False
    function = call.get("function") or {}
    if not isinstance(function, dict):
        return None, None, False
    name = function.get("name")
    ident = call.get("id")
    args = function.get("arguments")
    if not isinstance(name,str) or not name or not isinstance(ident,str) or not ident:
        return name, ident, False
    if isinstance(args,str):
        try:
            args=json.loads(args)
        except (ValueError,TypeError):
            return name,ident,False
    return name,ident,isinstance(args,dict)


def inspect(row):
    target=row.get("target")
    initial=row.get("messages")
    definitions=row.get("tools")
    if not isinstance(target,list) or not isinstance(initial,list) or not isinstance(definitions,list):
        raise ValueError("unexpected original NextSearch list schema")
    counts=Counter()
    calls={}
    seen=set()
    last_action_idx={}
    broken=False
    tool_error=0
    recovery_after_error=False
    obs_chars=0
    assistant_turns=0
    ordered_actions=[]
    for idx,msg in enumerate(target):
        if not isinstance(msg,dict):
            broken=True
            continue
        role=msg.get("role")
        counts["role_"+str(role)]+=1
        if role=="assistant":
            raw=msg.get("tool_calls") or []
            if raw:
                assistant_turns+=1
            for call in raw:
                name,ident,valid=_call(call)
                if not valid or ident in calls:
                    broken=True
                    continue
                calls[ident]=name
                last_action_idx[ident]=idx
                ordered_actions.append((idx,ident,name))
                counts["called_"+name]+=1
                if tool_error:
                    recovery_after_error=True
        elif role=="tool":
            ident=msg.get("tool_call_id")
            if not isinstance(ident,str) or ident not in calls or ident in seen:
                broken=True
                continue
            if idx<=last_action_idx[ident]:
                broken=True
                continue
            content=msg.get("content")
            if not isinstance(content,str) or not content.strip():
                broken=True
                continue
            seen.add(ident)
            obs_chars+=len(content)
            counts["nonempty_tool_feedback"]+=1
            if content.strip().casefold().startswith((
                "error:", "toolerror", "fetch failed", "search failed", "http error"
            )):
                tool_error+=1
        elif role not in ("system","user"):
            counts["unknown_role"]+=1
    # Trajectories that lack a final answer are not task-success teachers.
    last=target[-1] if target else {}
    last_is_final=(
        isinstance(last,dict)
        and last.get("role")=="assistant"
        and not (last.get("tool_calls") or [])
        and isinstance(last.get("content"),str)
        and bool(last["content"].strip())
    )
    gold=row.get("gold")
    raw_reference = gold.get("answer") if isinstance(gold,dict) else None
    if raw_reference is None:
        reference=""
    elif isinstance(raw_reference,(str,int,float)):
        reference=str(raw_reference).strip()
    else:
        # Ambiguous list/dict answer contracts are source-unscorable, never
        # silently converted into a Python repr as the 'correct' answer.
        reference=""
    source_reference_present=bool(reference)
    final=str(last.get("content","")).strip() if isinstance(last,dict) else ""
    exact=bool(source_reference_present and last_is_final and gaia_score(final,reference))
    complete=bool(calls and not broken and set(calls)==seen)
    # Search/fetch may be partly batched; still count *actual* observed calls
    # rather than simulated page traversals or assistant token count.
    grounded=bool(
        complete and len(calls)>=MIN_OBSERVED and assistant_turns>=3
        and counts["called_search"] and counts["called_fetch"]
    )
    return {
        "id":str(row.get("id") or ""),
        "license":str(row.get("license") or ""),
        "source":str(row.get("source") or ""),
        "band":str(row.get("band") or ""),
        "tool_names":sorted(set(calls.values())),
        "count_by_tool":dict(Counter(calls.values())),
        "original_declared_tools":sorted(str(t.get("name")) for t in definitions if isinstance(t,dict)),
        "assistant_action_turns":assistant_turns,
        "tool_actions":len(calls),
        "matched_nonempty_observations":len(seen),
        "observation_chars":obs_chars,
        "all_actual_calls_observed":complete,
        "strict_gold_answer_exact":exact,
        "source_reference_present":source_reference_present,
        "last_final_answer_present":last_is_final,
        "tool_error_count":tool_error,
        "recovery_after_error_observed":recovery_after_error,
        "long_search_fetch_candidate":grounded and exact,
        "tool_schema_native_to_our_gaia":False,
        "publisher_answer_provenance_only":True,
    }


def audit(rows,revision):
    if not rows:
        raise ValueError("empty publisher source")
    ids=set()
    long_unique_ids=set()
    summary=Counter()
    by_source=Counter()
    names=Counter()
    long_ids=[]
    bands=Counter()
    for row in rows:
        scored=inspect(row)
        ident=scored["id"]
        if not ident:
            raise ValueError("missing original NextSearch task identity")
        if ident in ids:
            # Original corpus contains multiple *distinct attempts* for the
            # same task ID (e.g. difficulty/failure bands). Not a corrupt
            # row, but a strict train/test/parent separation requirement.
            summary["duplicate_task_id_rows"]+=1
        ids.add(ident)
        if scored["license"].casefold()!="apache-2.0":
            raise ValueError("nonpermissive training license appears in Apache config")
        if "gaia" in ident.casefold() or "gaia" in scored["source"].casefold():
            raise ValueError("GAIA-source data forbidden for training")
        summary["total"]+=1
        summary["all_actions_have_actual_feedback"]+=scored["all_actual_calls_observed"]
        summary["original_final_present"]+=scored["last_final_answer_present"]
        summary["source_gold_exact"]+=scored["strict_gold_answer_exact"]
        summary["source_reference_missing"]+=int(not scored["source_reference_present"])
        summary["seven_or_more_real_tool_actions"]+=int(scored["tool_actions"]>=MIN_OBSERVED)
        summary["source_gold_exact_and_tool_feedback"]+=int(scored["strict_gold_answer_exact"] and scored["all_actual_calls_observed"])
        summary["long_search_fetch_gold_exact"]+=scored["long_search_fetch_candidate"]
        if scored["long_search_fetch_candidate"]:
            long_unique_ids.add(ident)
        summary["had_original_error_then_new_tool_action"]+=scored["recovery_after_error_observed"]
        summary["long_candidate_real_actions"]+=scored["tool_actions"] if scored["long_search_fetch_candidate"] else 0
        summary["long_candidate_observation_chars"]+=scored["observation_chars"] if scored["long_search_fetch_candidate"] else 0
        names.update(scored["count_by_tool"])
        by_source[scored["source"]]+=1
        bands[scored["band"]]+=1
        if scored["long_search_fetch_candidate"] and len(long_ids)<12:
            long_ids.append({
                "task_sha256":sha256(ident.encode()).hexdigest(),
                "source":scored["source"],
                "tool_actions":scored["tool_actions"],
                "tool_names":scored["tool_names"],
                "assistant_action_turns":scored["assistant_action_turns"],
            })
    return {
        "protocol":PROTOCOL,"source_repo":REPO,"config":CONFIG,
        "split":"train","source_revision":revision,
        "first_n_not_random":len(rows),"license":"apache-2.0",
        "stats":dict(summary),
        "unique_task_ids":len(ids),
        "unique_long_gold_exact_task_ids":len(long_unique_ids),
        "source_counts":dict(by_source),
        "band_counts":dict(bands),
        "declared_tools_seen":dict(names),
        "source_id_hashes_only":long_ids,
        "source_license_checked":True,
        "task_disjoint_with_gaia_source_name":True,
        "original_tool_messages_verbatim":True,
        "verified_success_from_our_agent":False,
        "independent_external_answer_review":False,
        "native_tool_contract_compatible":False,
        "training_data_exported":0,
        "gpu_optimizer_steps":0,
        "ready_for_qlora":False,
        "next_gate":"Verify source answer under same tool inputs and reward, compare native tool schemas; test closed-loop Qwen on a source-disjoint heldout cohort before any production adapter",
        "limits":"Original source train-only first-N sample, answer scored vs source-provided gold, live-web observations may become stale, synthetic/teacher-generated reasoning may not transfer; no GAIA questions used."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--limit",type=int,default=500)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.limit<=MAX_SOURCE_ROWS:
        raise ValueError("refuse overwrite or invalid limit")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision=HfApi().dataset_info(REPO).sha
    if revision != FROZEN_REVISION:
        raise ValueError("original source revision changed after prospective source audit")
    rows=list(islice(load_dataset(REPO, CONFIG,split="train",revision=revision,streaming=True),a.limit))
    if len(rows)!=a.limit:
        raise ValueError("original source was truncated")
    summary=audit(rows,revision)
    a.output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("NEXTSEARCH_APACHE_SOURCE_AUDIT",json.dumps(summary,sort_keys=True),flush=True)
    # HF streaming parquet can have a leftover downloader worker that aborts
    # Python during interpreter finalization after a valid report is written;
    # optional worker cleanup only runs after successful source validation.
    import os
    if os.environ.get("HF_STREAM_SAFE_EXIT")=="1":
        import sys
        with a.output.open("rb") as f:
            os.fsync(f.fileno())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__=="__main__":
    main()
