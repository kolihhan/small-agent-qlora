"""Freeze source-verified, GAIA-disjoint native search/read next actions for Qwen3.5-9B.

Uses only previously verified HotpotQA/2Wiki TRAIN trajectories. Never
translate foreign tool APIs, fabricate observations, or consume GAIA answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

SOURCES = {"2wikimultihopqa-train", "hotpotqa-distractor-train"}
NATIVE = {"search", "read", "inspect", "python"}
SEED = "native-qwen9b-source-nextactions-20261116-v1"


def function_name(call):
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        return None
    fn = call["function"]
    name, args = fn.get("name"), fn.get("arguments")
    if name not in NATIVE:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    if name == "search" and (not isinstance(args.get("query"), str)
                              or set(args) - {"query", "max_results"}):
        return None
    if name == "read" and (not isinstance(args.get("source"), str)
                            or set(args) - {"source", "offset_chars", "max_chars"}):
        return None
    return name


def observed_history(messages, target_index):
    pending, observations = set(), 0
    for msg in messages[:target_index]:
        if msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                if function_name(call) is None:
                    return False, observations
                cid = call.get("id")
                if not isinstance(cid, str) or not cid or cid in pending:
                    return False, observations
                pending.add(cid)
        elif msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid not in pending or not isinstance(msg.get("content"), str):
                return False, observations
            pending.remove(cid)
            observations += bool(msg["content"].strip())
    return not pending, observations


def eligible_actions(rows):
    candidates, seen = {"tool_call": [], "final": []}, set()
    for row in rows:
        if row.get("source") not in SOURCES or row.get("verified") is not True:
            continue
        task = row.get("task_id")
        if not isinstance(task, str) or not task or task in seen:
            raise ValueError("duplicate/missing original source task_id")
        seen.add(task)
        msgs = row.get("messages")
        indices = row.get("training_action_indices")
        tools = row.get("tools")
        if not isinstance(msgs, list) or not msgs or not isinstance(indices, list) or not indices or not isinstance(tools, list):
            raise ValueError("source missing curated real training actions/messages/tools")
        allowed = set()
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            if name not in NATIVE:
                allowed.clear()
                break
            allowed.add(name)
        if not allowed:
            continue
        for idx in indices:
            if type(idx) is not int or idx < 0 or idx >= len(msgs):
                raise ValueError("invalid curated original assistant index")
            target = msgs[idx]
            if target.get("role") != "assistant":
                raise ValueError("source target is not assistant")
            ok, observations = observed_history(msgs, idx)
            if not ok:
                continue
            calls = target.get("tool_calls") or []
            if calls:
                names = [function_name(c) for c in calls]
                if len(names) != 1 or names[0] not in ("search", "read") or names[0] not in allowed:
                    continue
                kind = "tool_call"
            else:
                if not str(target.get("content") or "").strip() or observations == 0:
                    continue
                kind = "final"
            order = sha256(f"{SEED}/{kind}/{row['source']}/{task}/{idx}".encode()).hexdigest()
            candidates[kind].append((order, row, idx, observations))
    return candidates


def freeze(rows, *, per_kind=24):
    choices = eligible_actions(rows)
    chosen, used = [], set()
    # Allocate the scarcer original action class first, maximizing parent-disjoint coverage.
    # This is source-only curation; no evaluation questions or GAIA labels are consulted.
    kinds = sorted(choices, key=lambda k: len({item[1]["task_id"] for item in choices[k]}))
    for kind in kinds:
        count = 0
        for _, row, idx, obs in sorted(choices[kind]):
            if row["task_id"] in used:
                continue
            chosen.append((row, idx, kind, obs))
            used.add(row["task_id"])
            count += 1
            if count == per_kind:
                break
        if count != per_kind:
            raise ValueError(f"insufficient parent-distinct native {kind} examples: {count}/{per_kind}")
    sources = Counter(row["source"] for row, _, _, _ in chosen)
    if len(sources) != 2 or min(sources.values()) < min(5, max(1, per_kind // 2)):
        raise ValueError("source diversity gate failed; never reduce it after selection")
    selected = []
    for row, idx, kind, obs in chosen:
        new = dict(row)
        new["training_action_indices"] = [idx]
        new["capability"] = "native_missing_evidence_continue" if kind == "tool_call" else "native_observed_evidence_finalize"
        new["source_native_actions_unchanged"] = True
        new["original_observations_before_action"] = obs
        selected.append(new)
    report = {
        "protocol": SEED,
        "base_model": "Qwen/Qwen3.5-9B",
        "sources": dict(sources),
        "selected_actions": {"tool_call": per_kind, "final": per_kind},
        "parent_distinct": len(used),
        "original_observations_before_actions": sum(obs for _, _, _, obs in chosen),
        "source_tool_schema_translations": 0,
        "GAIA_supervision": 0,
        "trained_adapter": False,
        "new_GAIA_score": None,
    }
    return selected, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite training freeze")
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    selected, report = freeze(rows)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    report["source_input_sha256"] = sha256(raw).hexdigest()
    report["train_rows_sha256"] = sha256(args.output.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("QWEN9B_NATIVE_SOURCE_ACTION_FREEZE", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
