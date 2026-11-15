"""Retrospective source-family shift: same frozen final renderer on untouched MuSiQue other8 model outputs.

No new model calls and no modification to the original Qwen native agent.
MuSiQue questions were independently selected before inference by an older
source archive; original 0/8 strict score is retained as the control.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from gaia_small_agent.benchmark.scoring import gaia_score
from scripts.pro_generic_final_render import render_final_answer

PROTOCOL = "generic-source-agnostic-final-render-MuSiQue-retrospective-v1"
SOURCE_PROTOCOL = "base-qwen9b-multihop-other8-read-path-failure-diagnostic-v1"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"


def replay(original: dict) -> dict:
    cases = original.get("cases")
    if (original.get("protocol") != SOURCE_PROTOCOL
        or original.get("total") != 8 or not isinstance(cases, list)
        or len(cases) != 8 or len({row["task_id"] for row in cases}) != 8
        or Counter(row["hop_count"] for row in cases) != Counter({3: 4, 4: 4})):
        raise ValueError("original eight unique source-driven MuSiQue model cases missing")
    results = []
    for row in cases:
        answer = str(row["answer"])
        gold = str(row["expected_answer"])
        original_exact = bool(row["correct"])
        completed = bool(row["completed"])
        if bool(completed and gaia_score(answer, gold)) != original_exact:
            raise ValueError("stored original MuSiQue strict correctness drift")
        new_answer, reason = render_final_answer(answer)
        current_exact = bool(completed and gaia_score(new_answer, gold))
        results.append({
            "task_id": row["task_id"], "hop_count": row["hop_count"],
            "before": original_exact, "after": current_exact,
            "renderer": reason, "changed": answer.strip() != new_answer,
            "raw_final": answer, "rendered_final": new_answer,
        })
    rescues = [r["task_id"] for r in results if r["after"] and not r["before"]]
    regressions = [r["task_id"] for r in results if r["before"] and not r["after"]]
    return {
        "protocol": PROTOCOL, "source_family": "MuSiQue 3/4-hop publisher TRAIN-derived source",
        "original_native_model_run": 43691026466,
        "source_protocol": SOURCE_PROTOCOL, "model_digest": DIGEST,
        "total": len(results), "before_exact": sum(r["before"] for r in results),
        "after_exact": sum(r["after"] for r in results),
        "changed": sum(r["changed"] for r in results),
        "render_reasons": dict(Counter(r["renderer"] for r in results)),
        "rescues": rescues, "regressions": regressions,
        "positive_retrospective_screen": len(rescues) >= 2 and not regressions,
        "model_calls": 0, "model_training": False, "promotion_allowed": False,
        "not_GAIA": True, "posthoc_original_model_run": True,
        "cases": results
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = replay(json.loads(args.source.read_text()))
    print("MUSIQUE_GENERIC_FINAL_RENDER_REPLAY", json.dumps(
        {k:v for k,v in result.items() if k != "cases"}, sort_keys=True), flush=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
