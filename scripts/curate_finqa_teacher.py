"""Curate only fully observed, independently numeric-correct FinQA read+python agents.

No fabricated action targets; no GAIA; no teacher corrections or dev labels.
At most one clean successful trace per distinct source parent task.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from gaia_small_agent.benchmark.scoring import gaia_score

SAMPLE = "::sample-"


def audit_verified(row: dict) -> list[int]:
    if (row.get("source") != "finqa-train"
        or row.get("verified") is not True
        or not isinstance(row.get("verification"), dict)):
        raise ValueError("source_or_verification")
    check = row["verification"]
    needed = ("completed", "final_answer_correct", "zero_tool_errors",
              "zero_duplicate_blocks", "required_tools_satisfied",
              "provenance_complete")
    if any(check.get(x) is not True for x in needed):
        raise ValueError("incomplete_verification")
    required = row.get("required_tools")
    if not isinstance(required, list) or not {"read","python"}.issubset(required):
        raise ValueError("missing_multitool_contract")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing_messages")
    users = [m for m in messages if isinstance(m,dict) and m.get("role")=="user"]
    if len(users) != 1:
        raise ValueError("ambiguous_user")
    if (not isinstance(messages[-1],dict)
        or messages[-1].get("role")!="assistant"
        or messages[-1].get("tool_calls")
        or not gaia_score(str(messages[-1].get("content") or ""),
                          str(row.get("expected_answer") or ""))):
        raise ValueError("final_answer_not_verified")

    indices, seen_ids, used_tools = [], set(), set()
    pos = 0
    while pos < len(messages):
        msg = messages[pos]
        if not isinstance(msg,dict):
            raise ValueError("malformed_message")
        if msg.get("role") == "tool":
            raise ValueError("orphan_tool_message")
        if msg.get("role") != "assistant":
            pos += 1
            continue
        calls = msg.get("tool_calls") or []
        indices.append(pos)
        if not calls:
            if pos != len(messages)-1:
                raise ValueError("early_final")
            pos += 1
            continue
        if not isinstance(calls,list):
            raise ValueError("malformed_tool_calls")
        after = messages[pos+1:pos+1+len(calls)]
        if len(after) != len(calls):
            raise ValueError("unpaired_tool")
        for c,m in zip(calls,after):
            fun=c.get("function") or {}
            name, cid = fun.get("name"), c.get("id")
            if not name or not cid or cid in seen_ids:
                raise ValueError("duplicate_or_bad_tool")
            seen_ids.add(cid)
            if (not isinstance(m,dict)
                or m.get("role")!="tool"
                or m.get("tool_call_id") != cid
                or m.get("name")!=name
                or str(m.get("content") or "").startswith("ToolError[")):
                raise ValueError("unverified_tool_result")
            if name=="read" and fun.get("arguments",{}).get("source") != "report.txt":
                raise ValueError("non_source_document_read")
            if name=="python" and not str(m.get("content") or "").strip():
                raise ValueError("empty_python_result")
            used_tools.add(name)
        pos+=1+len(calls)
    if not {"read","python"}.issubset(used_tools):
        raise ValueError("not_actually_multitool")
    if len(indices) < 3 or pos != len(messages):
        raise ValueError("missing_multi_step_trajectory")
    return indices


def curate(rows: list[dict]) -> tuple[list[dict],dict]:
    groups: dict[str,list[tuple[dict,list[int]]]] = defaultdict(list)
    rejected = Counter()
    seen_samples = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen_samples:
            raise ValueError("duplicate_observed_sample")
        seen_samples.add(task_id)
        try:
            indices = audit_verified(row)
            parent = task_id.split(SAMPLE,1)[0]
            if parent == task_id or not parent:
                raise ValueError("missing_sample_parent")
            groups[parent].append((row,indices))
        except (ValueError, TypeError, KeyError) as exc:
            rejected[str(exc)] += 1

    result = []
    questions = set()
    for parent, candidates in sorted(groups.items()):
        chosen,indices = min(candidates, key=lambda pair: (
            int(pair[0].get("metrics",{}).get("tool_calls", 999)),
            len(pair[1]), str(pair[0]["task_id"]),
        ))
        question = next(m["content"] for m in chosen["messages"] if m.get("role")=="user")
        if question in questions:
            rejected["duplicate_question"] += 1
            continue
        questions.add(question)
        item=dict(chosen)
        item["task_id"]=parent
        item["training_action_indices"]=indices
        item["selection_method"]="shortest_clean_observed_success"
        result.append(item)
    return result,{
        "source": "finqa-train", "input_verified_rollouts": len(rows),
        "accepted_unique_parents": len(result),
        "trainable_assistant_actions": sum(len(x["training_action_indices"]) for x in result),
        "rejected": dict(sorted(rejected.items())),
        "gaia_rows": 0, "training_ran": False,
        "boundary": "Success-only real read+python trajectories, not GAIA; source-diverse pilot not broad generalization proof",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",nargs="+",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.report.exists():
        raise FileExistsError("refuse overwrite")
    rows=[]
    for path in a.inputs:
        rows.extend(json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
    selected, report = curate(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in selected))
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("FINQA_CURATED_FINAL",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
