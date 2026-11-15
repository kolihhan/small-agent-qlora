"""Read-only, source-pinned original executed OpenHands trajectory readiness screen.

No training and no raw agent/user/source content exported. Original
publisher's success labels are NOT independently task-replayed; require
complete observed tool_call -> tool result links before counting an
assistant turn as usable supervision.

Real SWE execution is a domain-shift relative to GAIA and local ReadTool.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

SOURCE = "nebius/SWE-rebench-openhands-trajectories"
PROTOCOL = "nebius-executed-tool-call-success-source-readiness-v1"
MIN_ACTIONS = 7
MAX_LIMIT = 2000


def examine(row: dict) -> dict:
    """Pair executed actions; finish is the terminal unobserved OpenHands action.

    Real publisher protocol has exactly one assistant finish as its last
    message. No tool-role response for finish is expected. Every OTHER
    tool call must have an original, nonempty, later tool observation.
    """
    messages = row.get("trajectory") or []
    if not isinstance(messages, list):
        raise ValueError("unexpected published trajectory schema")
    pending, unique, seen, function_names = {}, set(), set(), Counter()
    all_ids = set()
    bad_args, missing_ids, bad_terminal = 0, 0, 0
    turns_with_tools, verified_results, response_chars = 0, 0, 0
    orphan, duplicated, out_of_order = 0, 0, 0
    terminal_finish = 0
    tool_errors = 0
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("publisher message must be dictionary")
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            active_actions = 0
            calls = message["tool_calls"]
            for call in calls:
                if not isinstance(call, dict):
                    bad_args += 1
                    continue
                callid = call.get("id")
                function = call.get("function") or {}
                name = function.get("name")
                args = function.get("arguments")
                if not isinstance(callid, str) or not callid:
                    missing_ids += 1
                    continue
                if callid in all_ids:
                    duplicated += 1
                    continue
                all_ids.add(callid)
                if not isinstance(name, str) or not name:
                    bad_args += 1
                    continue
                if isinstance(args, str):
                    try:
                        decoded = json.loads(args)
                        if not isinstance(decoded, dict):
                            bad_args += 1
                    except (ValueError, TypeError):
                        bad_args += 1
                elif not isinstance(args, dict):
                    bad_args += 1
                if name == "finish":
                    # Original OpenHands source emits a final finish tool
                    # call, without any subsequent tool reply. Only one
                    # final assistant message may use this exception.
                    terminal_finish += 1
                    if idx != len(messages) - 1 or len(calls) != 1:
                        bad_terminal += 1
                    continue
                unique.add(callid)
                pending[callid] = idx
                function_names[name] += 1
                active_actions += 1
            turns_with_tools += int(active_actions > 0)
        elif role == "tool":
            callid = message.get("tool_call_id")
            if not isinstance(callid, str) or not callid or callid not in pending:
                orphan += 1
                continue
            if callid in seen:
                duplicated += 1
                continue
            if pending[callid] >= idx:
                out_of_order += 1
                continue
            body = message.get("content")
            if not isinstance(body, str) or not body.strip():
                continue
            seen.add(callid)
            verified_results += 1
            response_chars += len(body)
            if "error" in body[:250].casefold() or "traceback" in body[:250].casefold():
                tool_errors += 1
    labeled_success = row.get("resolved") == 1 and row.get("exit_status") == "submit"
    complete = (
        bool(unique) and len(seen) == len(unique)
        and terminal_finish == 1 and bad_terminal == 0
        and bad_args == 0 and missing_ids == 0
        and orphan == 0 and duplicated == 0 and out_of_order == 0
    )
    good = bool(
        labeled_success and complete and len(unique) >= MIN_ACTIONS
        and turns_with_tools >= 3 and str(row.get("model_patch") or "").strip()
    )
    return {
        "success_label": labeled_success,
        "resolved": row.get("resolved"),
        "tool_calls": len(unique),
        "paired_tool_observations": verified_results,
        "terminal_finish_without_response": terminal_finish,
        "terminal_finish_valid": terminal_finish == 1 and bad_terminal == 0,
        "assistant_tool_turns": turns_with_tools,
        "all_calls_observed": complete,
        "tool_errors_heuristic": tool_errors,
        "total_observation_chars": response_chars,
        "bad_arguments": bad_args,
        "orphan_observations": orphan,
        "duplicate_call_ids": duplicated,
        "out_of_order_observations": out_of_order,
        "function_names": dict(function_names),
        "long_executed_success_candidate": good,
        "success_independently_replayed": False,
        "training_ready_for_native_gaia_policy": False,
    }


def aggregate(rows: list[dict], revision: str) -> dict:
    if not rows:
        raise ValueError("no publisher records")
    stats=Counter()
    actions=Counter()
    good_repos=set()
    good_issue_ids=set()
    seen_trajectory_ids=set()
    seen_issue_ids=Counter()
    example_ids=[]
    for r in rows:
        identity=str(r.get("trajectory_id") or "")
        if not identity or identity in seen_trajectory_ids:
            raise ValueError("repeated/missing original trajectory_id")
        seen_trajectory_ids.add(identity)
        instance=str(r.get("instance_id") or "")
        if not instance:
            raise ValueError("missing source instance_id")
        seen_issue_ids[instance]+=1
        item=examine(r)
        stats["total"]+=1
        stats["labeled_success"]+=int(item["success_label"])
        stats["complete_observation_pairing"]+=int(item["all_calls_observed"])
        stats["success_complete_pairing"]+=int(item["success_label"] and item["all_calls_observed"])
        stats["success_at_least_seven_calls"]+=int(item["success_label"] and item["tool_calls"]>=MIN_ACTIONS)
        stats["success_complete_seven_calls_three_turns"]+=int(item["long_executed_success_candidate"])
        stats["with_any_tool_calls"]+=int(item["tool_calls"]>0)
        stats["with_heuristic_tool_error"]+=int(item["tool_errors_heuristic"]>0)
        stats["bad_tool_arg_rows"]+=int(item["bad_arguments"]>0)
        stats["orphan_tool_observations_rows"]+=int(item["orphan_observations"]>0)
        stats["completed_without_patch"]+=int(item["success_label"] and not str(r.get("model_patch") or "").strip())
        if item["long_executed_success_candidate"]:
            good_repos.add(str(r.get("repo") or ""))
            good_issue_ids.add(instance)
            stats["long_candidate_tool_calls"]+=item["tool_calls"]
            stats["long_candidate_tool_turns"]+=item["assistant_tool_turns"]
            stats["long_candidate_tool_output_chars"]+=item["total_observation_chars"]
            actions.update(item["function_names"])
            if len(example_ids)<10:
                example_ids.append({
                    "trajectory_sha256":sha256(identity.encode()).hexdigest(),
                    "issue_sha256":sha256(instance.encode()).hexdigest(),
                    "tool_calls":item["tool_calls"],
                    "tool_turns":item["assistant_tool_turns"],
                    "tool_names":sorted(item["function_names"]),
                })
    return {
        "protocol":PROTOCOL,"original_dataset":SOURCE,"dataset_revision":revision,
        "source_license":"CC-BY-4.0 per publisher metadata",
        "source_type":"actual OpenHands agent executions on public GitHub issues",
        "source_task_family":"SWE-rebench coding agents, NOT GAIA",
        "publisher_model":"Qwen3-Coder-480B-A35B-Instruct",
        "source_rows_first_n":len(rows),"source_order_not_random":True,
        "scoring_rule":"publisher resolved=1 AND exit_status=submit, no independent replay",
        "minimum_observed_tool_actions":MIN_ACTIONS,
        "minimum_assistant_tool_turns":3,
        "stats":dict(stats),
        "successful_long_unique_issues":len(good_issue_ids),
        "successful_long_distinct_repos":len(good_repos),
        "observed_tool_names":dict(actions),
        "source_example_id_hashes_only":example_ids,
        "unverified_success_claims_prohibited":True,
        "tool_schema_mismatch_with_gaia":True,
        "native_training_examples_exported":0,
        "gpu_training_steps":0,
        "ready_for_qlora":False,
        "next_gate":"Source-local rewards verified enough for SFT pilot AND map OpenHands tool/observation surface to original agent; then matched source-disjoint autonomous Qwen vs v3B",
        "limits":"Limited original dataset order sample; publisher labels not independently replayed; OpenHands SWE distribution may not transfer to GAIA."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--limit",type=int,default=500)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.limit<=MAX_LIMIT:
        raise ValueError("invalid input/output/limit")
    from huggingface_hub import HfApi
    from datasets import load_dataset
    revision=HfApi().dataset_info(SOURCE).sha
    rows=list(islice(load_dataset(
        SOURCE,split="train",streaming=True,revision=revision
    ),a.limit))
    if len(rows)!=a.limit:
        raise ValueError("publisher source unexpectedly truncated")
    result=aggregate(rows,revision)
    print("NEBIUS_REAL_EXECUTED_LONG_TRAJECTORIES",json.dumps(result,sort_keys=True),flush=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
