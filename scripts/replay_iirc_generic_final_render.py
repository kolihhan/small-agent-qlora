"""Original IIRC DEV source-family shift test for frozen generic final renderer.

Consumes original *already completed* Qwen9B autonomous closed-loop output
from 2026-11-15 PR #41. Does not rerun a model. The generic renderer must be
frozen before the IIRC final answers are inspected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from gaia_small_agent.benchmark.scoring import gaia_score
from scripts.pro_generic_final_render import render_final_answer

PROTOCOL = "generic-source-agnostic-final-render-IIRC-retrospective-v1"
ORIGINAL_PROTOCOL = "iirc-dev-source-links-8-v1"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
ARMS = ("flat", "linked")


def replay(previous: dict) -> dict:
    cases = previous.get("cases")
    if (previous.get("protocol") != ORIGINAL_PROTOCOL
        or previous.get("digest") != DIGEST or not isinstance(cases, list)
        or len(cases) != 8 or len({row["id"] for row in cases}) != 8
        or len({row["page"] for row in cases}) != 8):
        raise ValueError("original independent IIRC source/identity/model mismatch")
    records = []
    for record in cases:
        out = {"id": record["id"], "page": record["page"]}
        reference = str(record["reference"])
        for arm in ARMS:
            raw = str(record[arm]["answer"])
            previous_exact = bool(record[arm]["exact"])
            if bool(record[arm]["completed"] and gaia_score(raw, reference)) != previous_exact:
                raise ValueError(f"original IIRC exact scoring drift on {record['id']} / {arm}")
            rendered, reason = render_final_answer(raw)
            now = bool(record[arm]["completed"] and gaia_score(rendered, reference))
            out[arm] = {
                "raw_exact": previous_exact, "rendered_exact": now,
                "renderer": reason, "changed": rendered != raw.strip(),
                "completed": bool(record[arm]["completed"]),
                "raw_final": raw, "rendered_final": rendered
            }
        records.append(out)
    scores = {arm: {
        "raw_exact": sum(r[arm]["raw_exact"] for r in records),
        "rendered_exact": sum(r[arm]["rendered_exact"] for r in records),
        "changed": sum(r[arm]["changed"] for r in records),
        "render_reasons": dict(Counter(r[arm]["renderer"] for r in records)),
    } for arm in ARMS}
    rescues = [
        f"{r['id']}:{arm}" for r in records for arm in ARMS
        if not r[arm]["raw_exact"] and r[arm]["rendered_exact"]
    ]
    regressions = [
        f"{r['id']}:{arm}" for r in records for arm in ARMS
        if r[arm]["raw_exact"] and not r[arm]["rendered_exact"]
    ]
    passing = (
        len(rescues) - len(regressions) >= 2 and not regressions
        and len({x.split(":")[0] for x in rescues}) >= 2
    )
    return {
        "protocol": PROTOCOL, "original_protocol": ORIGINAL_PROTOCOL,
        "source_family": "IIRC publisher original DEV sources",
        "historical_run": 43700029046, "tasks": len(records),
        "model_digest": DIGEST, "arms": scores, "rescues": rescues,
        "regressions": regressions, "positive_retrospective_screen": passing,
        "promotion_allowed": False, "model_calls": 0, "model_training": False,
        "posthoc_original_model_runs": True,
        "caveat": "IIRC is a different source family from where final-answer renderer was designed, but these are stored previously exposed model finals, not a new independent live trial. A positive retrospective outcome is not a QLoRA or GAIA improvement.",
        "cases": records
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = replay(json.loads(args.source.read_text()))
    print("IIRC_GENERIC_FINAL_RENDER_REPLAY", json.dumps(
        {k:v for k,v in result.items() if k != "cases"}, sort_keys=True), flush=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
