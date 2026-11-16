"""Constrained-terminal safety probe on predeclared previously-correct step12 GAIA cases.

Both task hashes come from frozen, previously exposed GAIA100 Qwen3.5-9B v7
original-source physical run 43865621132 (shards 19 and 2). They each had
original.correct=True AND 11 tool calls, hence reached forced final step12.
No use of reference answers in prompting, no new training or model change.

Each task is run ONCE to obtain original preterminal tool transcript. Only
the exact captured final request is replayed with additional JSON Schema.
A previous-correct task that now stops early or becomes wrong is *not*
counted as a successful decoder safety test. This is exposed-data diagnostic.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.benchmark.gaia100 import (
    GAIA_REVISION,LOCAL_PROTOCOL_SEED,load_validation,select_gaia_partition,
)
from scripts.gaia_json_schema_prefix_replay8 import run_case

SOURCE_RUN=43865621132
MODEL="same fixed Qwen3.5-9B v7 adapter"
CONTROL_TASK_HASHES=(
    "3f9b8f9b1b73a22d51c1feac23b6c6328a5019597fdfefda83d561e5c6dc6fdd",
    "e424150204292b186aab0dc10924881308dc81e098b971cbf1758a4290bd9ffb",
)
N=2
PROTOCOL="GAIA-PREVIOUSLY-CORRECT-STEP12-EXACT-PREFIX-CONTROLS2-V1"


def select_source_rows() -> list[dict]:
    source=select_gaia_partition(load_validation(),partition="evaluation",seed=LOCAL_PROTOCOL_SEED)
    if len(source)!=100:
        raise RuntimeError("Original frozen GAIA100 source changed")
    lookup={sha256(str(x["task_id"]).encode()).hexdigest():x for x in source}
    if len(lookup)!=100 or not set(CONTROL_TASK_HASHES).issubset(lookup):
        raise RuntimeError("Previously observed correct-terminal cases not in original evaluation100")
    rows=[lookup[k] for k in CONTROL_TASK_HASHES]
    if any(int(r["Level"])!=2 for r in rows):
        raise RuntimeError("Originally predeclared Level2 controls changed")
    return rows


def run_controls() -> dict:
    rows=select_source_rows()
    cases=[run_case(r) for r in rows]
    if [x["task_hash"] for x in cases]!=list(CONTROL_TASK_HASHES):
        raise RuntimeError("Control task hash order drift")
    for case in cases:
        a,b=case["original"],case["json_schema_final"]
        if a["tool_calls"]!=b["tool_calls"] or a["tool_errors"]!=b["tool_errors"]:
            raise RuntimeError("Identical preterminal evidence and tools violated")
        if case["terminal_reached"] and not case.get("exact_prefix_sha256"):
            raise RuntimeError("Exact original terminal messages were not captured")
    reached=[c for c in cases if c["terminal_reached"]]
    valid_safety=[c for c in reached if c["original"]["correct"]]
    regress=[c for c in valid_safety if not c["json_schema_final"]["correct"]]
    result={
        "status":"PRIOR_CORRECT_STEP12_EXPOSED_CONTROL_SCREEN_COMPLETE",
        "protocol":PROTOCOL,"original_frozen_run_id":SOURCE_RUN,
        "source_revision":GAIA_REVISION,
        "model":MODEL,
        "prior_original_correct_and_11_tool_calls":len(CONTROL_TASK_HASHES),
        "n":N,"terminal_reached_now":len(reached),
        "original_still_correct_at_terminal_now":len(valid_safety),
        "terminal_correct_regressions":len(regress),
        "original_correct_now":sum(c["original"]["correct"] for c in cases),
        "json_correct_now":sum(c["json_schema_final"]["correct"] for c in cases),
        "original_completed_now":sum(c["original"]["completed"] for c in cases),
        "json_completed_now":sum(c["json_schema_final"]["completed"] for c in cases),
        "tested_terminal_correct_original_task_hashes":[c["task_hash"] for c in valid_safety],
        "regressed_task_hashes":[c["task_hash"] for c in regress],
        "preterminal_replay_identity_verified":True,
        "new_model_training":False,"new_hidden_benchmark":False,
        "no_promotion":True,
        "claim_boundary":"Controls are two previously seen source cases selected for correct original forced-terminal outcome, not independent or unbiased. If neither still reaches the original-correct forced-terminal state, decoder safety remains unmeasured. Any observed regression rejects broad promotion; zero regressions in N<=2 does not establish safety.",
        "cases":cases,
    }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report=run_controls()
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("GAIA_TERMINAL_CORRECT_CONTROL2_FINAL",
          json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
