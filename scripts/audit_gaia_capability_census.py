"""CPU-only GAIA100 unsupported-modality failure census from frozen paired traces.

An attachment's media type is a possibility signal, NOT causal proof that a new
image/audio/video tool would rescue a failed task. Never export task content.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_v8_gaia100_paired_regressions import arm_rows

KNOWN_TAGS = (
    "semantic_image_unsupported",
    "semantic_audio_unsupported",
    "semantic_video_unsupported",
)
KNOWN_READ_ERRORS = {"UNSUPPORTED_BINARY", "MISSING_DEPENDENCY", "CONTENT_TOO_LARGE"}


def evidence_for(row):
    events = row.get("trace") or []
    calls = {str((e.get("data") or {}).get("name"))
             for e in events if e.get("kind") == "tool_call"}
    binary_errors = {
        str((e.get("data") or {}).get("error_code"))
        for e in events
        if e.get("kind") == "tool_result"
        and (e.get("data") or {}).get("ok") is False
    } & KNOWN_READ_ERRORS
    return {"read_attempted": "read" in calls,
            "inspect_attempted": "inspect" in calls,
            "known_read_errors": sorted(binary_errors)}


def category(row):
    tag = row.get("capability_gap")
    if tag not in KNOWN_TAGS and tag is not None:
        raise ValueError("unexpected capability tag")
    return tag or "no_tagged_gap"


def summarize_arm(rows):
    buckets = defaultdict(list)
    for row in rows.values():
        buckets[category(row)].append(row)
    result = {}
    for tag in ("no_tagged_gap",) + KNOWN_TAGS:
        part = buckets[tag]
        wrong = [r for r in part if not bool(r["correct"])]
        errors = Counter(
            code
            for r in wrong
            for code in evidence_for(r)["known_read_errors"]
        )
        result[tag] = {
            "tasks": len(part),
            "correct": sum(bool(r["correct"]) for r in part),
            "wrong": len(wrong),
            "completed_wrong": sum(bool(r["completed"]) for r in wrong),
            "wrong_with_read_attempt": sum(evidence_for(r)["read_attempted"] for r in wrong),
            "wrong_with_inspect_attempt": sum(evidence_for(r)["inspect_attempted"] for r in wrong),
            "wrong_with_known_read_error": sum(bool(evidence_for(r)["known_read_errors"]) for r in wrong),
            "known_read_errors": dict(sorted(errors.items())),
        }
    return result


def analyze(base_rows, adapter_rows):
    assert len(base_rows) == len(adapter_rows) == 100
    assert set(base_rows) == set(adapter_rows)
    outcome_by_gap = defaultdict(Counter)
    for tid in base_rows:
        a, b = base_rows[tid], adapter_rows[tid]
        if category(a) != category(b):
            raise ValueError("source attachment category changed between paired runs")
        outcome = ("both_correct" if a["correct"] and b["correct"] else
                   "base_only" if a["correct"] else
                   "adapter_only" if b["correct"] else "both_wrong")
        outcome_by_gap[category(a)][outcome] += 1
    base_stats = summarize_arm(base_rows)
    adapter_stats = summarize_arm(adapter_rows)
    return {
        "protocol": "GAIA100-CAPABILITY-CENSUS-20261123-V1",
        "original_paired_run": 44980055775,
        "n_paired_tasks": 100,
        "no_model_inference": True,
        "no_training": True,
        "no_source_task_or_trace_export": True,
        "base": base_stats,
        "qlora_v8": adapter_stats,
        "outcome_by_attachment_category": {
            k: dict(sorted(v.items())) for k,v in sorted(outcome_by_gap.items())
        },
        "verified_limits": [
            "Type tags identify an attachment modality, not causal necessity",
            "Tool UNSUPPORTED_BINARY proves a tool read failure only, not a rescuable answer",
            "No image/audio/video intervention has been evaluated; gains cannot be claimed",
            "Original public GAIA100 may contain retrieved benchmark answers; do not promote scores",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = arm_rows(args.root, "base", 20)
    adapter = arm_rows(args.root, "qlora-v8", 20)
    report = analyze(base, adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print("GAIA100_CAPABILITY_CENSUS", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
