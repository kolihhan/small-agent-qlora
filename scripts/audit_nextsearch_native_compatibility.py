"""Gold-blind compatibility gate for source NextSearch QLoRA -> native GAIA tools.

Only measures whether original source actions could be *expressed* with the
unchanged native tool JSON schema. It DOES NOT relabel historical observations,
declare teacher actions optimal, or generate any training examples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse

FROZEN_48_SHA = "5420cfdb6111806a23b4bf6cb9cf04c77c75b5f0cf4af6c0b63727f312f40432"
STATES = {"evidence_missing_continue": 24, "evidence_present_finalize": 24}


def _arguments(call: dict) -> tuple[str, dict]:
    fn = call.get("function")
    if not isinstance(fn, dict):
        return "", {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return "", {}
    return str(fn.get("name") or ""), args if isinstance(args, dict) else {}


def action_compatibility(call: dict) -> str:
    """Categorize schema feasibility, never certify source-observation parity."""
    name, args = _arguments(call)
    if name == "search":
        if set(args) <= {"query", "max_results"} and isinstance(args.get("query"), str) and args["query"].strip():
            return "native_exact_schema"
        if set(args) <= {"objective", "search_queries"}:
            qs = args.get("search_queries")
            if isinstance(qs, list) and len(qs) == 1 and isinstance(qs[0], str) and qs[0].strip():
                return "one_query_syntax_only"
            if isinstance(qs, list) and len(qs) > 1:
                return "batched_search"
    if name == "read":
        if set(args) <= {"source", "offset_chars", "max_chars"} and isinstance(args.get("source"), str) and args["source"].strip():
            return "native_exact_schema"
    if name == "fetch" and set(args) <= {"objective", "urls", "offset"}:
        urls = args.get("urls")
        if isinstance(urls, list) and len(urls) > 1:
            return "batched_fetch"
        if isinstance(urls, list) and len(urls) == 1 and isinstance(urls[0], str):
            parsed = urlparse(urls[0])
            if parsed.scheme in ("http", "https") and parsed.netloc:
                # Source 'offset' semantics and text observations have not been
                # reproduced with native ReadTool. Never call this native-ready.
                if args.get("offset", 0) == 0:
                    return "one_url_syntax_only"
                return "fetch_offset_not_proven_equivalent"
    return "not_native_compatible"


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no source training rows")
    states = Counter()
    classes = Counter()
    history = Counter()
    parent_ids = set()
    task_ids = set()
    for row in rows:
        state = row.get("capability")
        if state not in STATES or row.get("source_split") != "train":
            raise ValueError("unexpected state or protected source split")
        if row.get("original_teacher_action_unchanged") is not True or row.get("task_success_independently_replayed") is not False:
            raise ValueError("non-original or oracle-validated training label")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in task_ids:
            raise ValueError("missing or duplicated source action identity")
        task_ids.add(task_id)
        parent_ids.add(row.get("source_parent_sha256"))
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or row.get("training_action_indices") != [len(messages) - 1]:
            raise ValueError("source next-action/history contract changed")
        target = messages[-1]
        if target.get("role") != "assistant":
            raise ValueError("target is not an original assistant action")
        states[state] += 1
        calls = target.get("tool_calls") or []
        if state == "evidence_present_finalize":
            if calls or not str(target.get("content") or "").strip():
                raise ValueError("complete-evidence state lacks final")
            classes["final_action"] += 1
        elif len(calls) != 1:
            classes["multi_or_missing_target_calls"] += 1
        else:
            classes[action_compatibility(calls[0])] += 1

        old_calls = [
            call
            for msg in messages[:-1]
            if msg.get("role") == "assistant"
            for call in (msg.get("tool_calls") or [])
        ]
        history["prior_tool_calls"] += len(old_calls)
        if old_calls and all(action_compatibility(c) == "native_exact_schema" for c in old_calls):
            history["fully_native_schema_history"] += 1
        history["rows_with_prior_source_calls"] += bool(old_calls)
    return {
        "source_actions": len(rows),
        "parent_tasks": len(parent_ids),
        "states": dict(states),
        "target_actions": dict(classes),
        "history": dict(history),
        "source_tool_observations_replayed_with_native_runtime": 0,
        "native_qlora_training_examples_certified": 0,
        "new_adapter_training": False,
        "new_gaia_score": None,
        "decision": "Do not reuse old source tool observations as native GAIA SFT. Collect genuine native search/read histories before training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if sha256(raw).hexdigest() != FROZEN_48_SHA:
        raise ValueError("original frozen 48-example source hash changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    report = summarize(rows)
    if len(rows) != 48 or report["states"] != STATES or report["parent_tasks"] != 28:
        raise ValueError("original 48/24:24/28 training split changed")
    if args.output.exists():
        raise FileExistsError("do not overwrite source audit evidence")
    report["source_sha256"] = FROZEN_48_SHA
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("ORIGINAL_NEXTSEARCH_TO_NATIVE_GAIA_COMPATIBILITY", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
