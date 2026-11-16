"""Compare real SFT target actions against the *actual* GAIA Agent tool contracts.

No model calls or private questions. Operates on original immutable source
artifacts; publisher 'verified' does not imply native policy compatibility.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from gaia_small_agent.tools import default_tools

_ERROR_PATTERN = re.compile(r"(?im)^\s*(?:tool error\s*:|ToolError\[)")
EXPECTED_SHA = {
    "nextsearch": "5420cfdb6111806a23b4bf6cb9cf04c77c75b5f0cf4af6c0b63727f312f40432",
    "native": "c900850f676834a5db295ae6a9b9f5b1ccbf5956dc78fdfef2f18dcb560c3835",
}


def load_jsonl(path: Path, *, expected_sha: str) -> list[dict]:
    content = path.read_bytes()
    actual = sha256(content).hexdigest()
    if actual != expected_sha:
        raise ValueError("Original source SHA mismatch")
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Empty or malformed original SFT source")
    ids = [row.get("task_id") for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate or invalid source task IDs")
    return rows


def _target_actions(row: dict) -> list[dict]:
    messages = row.get("messages")
    indexes = row.get("training_action_indices")
    if not isinstance(messages, list) or not isinstance(indexes, list) or not indexes:
        raise ValueError("Training row has missing messages or action indices")
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(messages) for i in indexes):
        raise ValueError("Training row index not within original history")
    if indexes != sorted(set(indexes)):
        raise ValueError("Action indices must be sorted and unique")
    actions = [messages[i] for i in indexes]
    if any(action.get("role") != "assistant" for action in actions):
        raise ValueError("Supervised target must be an assistant turn")
    return actions


def _tool_error_observed(row: dict) -> bool:
    return any(
        message.get("role") == "tool" and _ERROR_PATTERN.search(str(message.get("content") or ""))
        for message in row["messages"]
    )


def audit_rows(rows: list[dict], *, source_label: str, tool_registry=None) -> dict:
    native = tool_registry or {tool.name: tool for tool in default_tools()}
    counts = Counter()
    source_counter = Counter()
    mismatch_names = Counter()
    shape_errors = Counter()
    parents_with_observed_error = set()
    for row in rows:
        parent = str(row.get("source_parent_task_id") or row["task_id"])
        source_counter[parent] += 1
        selected = _target_actions(row)
        counts["target_actions"] += len(selected)
        counts["selected_final_actions"] += sum(not bool(act.get("tool_calls")) for act in selected)
        if any(bool(a.get("tool_calls")) for a in selected):
            counts["target_tool_action_windows"] += 1
        for act in selected:
            for call in act.get("tool_calls") or []:
                f = call.get("function") or {}
                name = f.get("name")
                arguments = f.get("arguments")
                counts["target_tool_calls"] += 1
                if name not in native:
                    counts["unknown_native_tool_calls"] += 1
                    mismatch_names[str(name)] += 1
                    continue
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = None
                if not isinstance(arguments, dict):
                    counts["invalid_native_arguments"] += 1
                    shape_errors[str(name)] += 1
                    continue
                error = native[name].validate_arguments(arguments)
                if error:
                    counts["invalid_native_arguments"] += 1
                    shape_errors[str(name)] += 1
                else:
                    counts["valid_native_tool_calls"] += 1

        has_observed_error = _tool_error_observed(row)
        if has_observed_error:
            counts["observed_tool_error_windows"] += 1
            parents_with_observed_error.add(parent)
            if row.get("verification", {}).get("zero_tool_errors") is True:
                counts["metadata_zero_error_contradictions"] += 1
            if row.get("metrics", {}).get("tool_errors") == 0:
                counts["metrics_zero_error_contradictions"] += 1
        if row.get("source_context_is_partial") is True:
            counts["partial_source_context_windows"] += 1
        if row.get("task_success_independently_replayed") is True:
            counts["independently_replayed_task_success_windows"] += 1
        if row.get("training_ready_for_production") is True:
            counts["source_claims_production_ready"] += 1

    if counts["target_tool_calls"] != (counts["valid_native_tool_calls"] +
                                     counts["invalid_native_arguments"] +
                                     counts["unknown_native_tool_calls"]):
        raise AssertionError("Incomplete target tool-call classification")
    return {
        "source": source_label,
        "total_windows": len(rows),
        "distinct_parents": len(source_counter),
        "native_tool_names": sorted(native),
        "target_actions": counts["target_actions"],
        "selected_final_actions": counts["selected_final_actions"],
        "target_tool_action_windows": counts["target_tool_action_windows"],
        "target_tool_calls": counts["target_tool_calls"],
        "valid_native_tool_calls": counts["valid_native_tool_calls"],
        "invalid_native_arguments": counts["invalid_native_arguments"],
        "unknown_native_tool_calls": counts["unknown_native_tool_calls"],
        "invalid_arguments_by_tool": dict(sorted(shape_errors.items())),
        "unknown_names": dict(sorted(mismatch_names.items())),
        "observed_tool_error_windows": counts["observed_tool_error_windows"],
        "affected_error_parents": len(parents_with_observed_error),
        "metadata_zero_error_contradictions": counts["metadata_zero_error_contradictions"],
        "metrics_zero_error_contradictions": counts["metrics_zero_error_contradictions"],
        "partial_source_context_windows": counts["partial_source_context_windows"],
        "independently_replayed_task_success_windows": counts["independently_replayed_task_success_windows"],
        "source_claims_production_ready": counts["source_claims_production_ready"],
        "decision": "FAIL_NATIVE_TOOL_CONTRACT" if (counts["invalid_native_arguments"] or counts["unknown_native_tool_calls"])
                    else "PASS_NATIVE_SCHEMA_ONLY_NOT_TASK_CORRECTNESS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nextsearch", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = load_jsonl(args.nextsearch, expected_sha=EXPECTED_SHA["nextsearch"])
    q = load_jsonl(args.native, expected_sha=EXPECTED_SHA["native"])
    if len(n) != 48 or len(q) != 40:
        raise ValueError("Expected the 48+40 immutable action rows, no re-sampling")
    report = {
        "protocol": "source-native-target-action-compatibility-v1",
        "source_sha256": EXPECTED_SHA,
        "model_base": "Qwen3.5-9B, unchanged",
        "nextsearch": audit_rows(n, source_label="NextSearch48 original publisher tool surface"),
        "native": audit_rows(q, source_label="HotpotQA26+2Wiki14 native read-only source"),
        "limitations": [
            "Tool schema compatibility is necessary but not sufficient for correct agent decisions.",
            "Tool error scan recognizes explicit tool-error markers, not every silent partial failure.",
            "Publisher final answers are not independently replayed model task successes.",
            "No GAIA task inference, training or new score was performed.",
        ],
    }
    if report["nextsearch"]["decision"] != "FAIL_NATIVE_TOOL_CONTRACT":
        raise ValueError("NextSearch source expectation changed; investigation required")
    if report["native"]["decision"] != "PASS_NATIVE_SCHEMA_ONLY_NOT_TASK_CORRECTNESS":
        raise ValueError("Native source unexpectedly incompatible with original runtime")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
