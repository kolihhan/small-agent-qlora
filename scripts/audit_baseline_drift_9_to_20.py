"""Provenance-only historical GAIA Base drift audit. Does not claim causal identification."""
from __future__ import annotations
from collections import Counter
import glob
import json
from pathlib import Path
import re

RUN_V3 = 38911211243
RUN_V8 = 44980055775
V3_BASE = {"correct": 9, "completed": 45, "invalid_final": 42, "tool_calls": 764}
EXPECTED_SELECTION_SHA = "22ce65f410d45e2e84af5a0ad1eafeadb311b08bb1234c968a114174c29bb373"
EXPECTED_FROZEN_COMMIT = "08fbcaa5ed82c5238188c45d5748993bae9bf1e1"
EXPECTED_BASE_MODEL_SHA = "8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
EXPECTED_LLAMA_SHA = "36ca310be4405acb7ba59b32dbdce32bb9cd7b1ef95358997b7eaed2c6f0188b"

def assert_source_workflows_match():
    old = Path(".github/workflows/gaia-v3d-clean-same-runner.yml").read_text()
    new = Path(".github/workflows/qlora-v8-gaia100-auto.yml").read_text()
    for name, expected in (
        ("FROZEN_EVAL_COMMIT", EXPECTED_FROZEN_COMMIT),
        ("MODEL_SHA256", EXPECTED_BASE_MODEL_SHA),
        ("LLAMA_RELEASE_SHA256", EXPECTED_LLAMA_SHA),
    ):
        def get(content):
            m = re.search(r"^\s*" + re.escape(name) + r":\s*[\"']?([0-9a-f]+)", content, flags=re.M)
            if m is None:
                raise ValueError("missing original workflow provenance: " + name)
            return m.group(1)
        if not (get(old) == get(new) == expected):
            raise ValueError("cross-run frozen configuration mismatch: " + name)

def audit(root: Path):
    assert_source_workflows_match()
    result = {}
    hashes = set()
    shards = set()
    for arm in ("base", "qlora-v8"):
        rows = {}
        paths = sorted(root.glob("gaia-v8-paired-shard-*/" + arm + "/shard-*/results.jsonl"))
        if len(paths) != 20:
            raise ValueError("not all twenty original GAIA shard results are available")
        for p in paths:
            cfg = json.loads((p.parent / "run_config.json").read_text())
            hashes.add(cfg["full_selection_sha256"])
            shard = int(cfg["shard_index"])
            shards.add(shard)
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = row["task_id"]
                if tid in rows:
                    raise ValueError("duplicate task id")
                rows[tid] = row
        if len(rows) != 100:
            raise ValueError("not exactly 100 original evaluation task IDs")
        result[arm] = {
            "n": 100,
            "correct": sum(r["correct"] is True for r in rows.values()),
            "completed": sum(r["completed"] is True for r in rows.values()),
            "invalid_final": sum(r["stop_reason"] == "invalid_final" for r in rows.values()),
            "tool_calls": sum(int(r["tool_calls"]) for r in rows.values()),
        }
        if arm == "base":
            base_ids = set(rows)
        elif set(rows) != base_ids:
            raise ValueError("Base and QLoRA v8 compared different task IDs")
    if hashes != {EXPECTED_SELECTION_SHA} or shards != set(range(20)):
        raise ValueError("GAIA question selection hash or shard coverage changed")
    if (result["base"]["correct"], result["qlora-v8"]["correct"]) != (20, 21):
        raise ValueError("latest archived GAIA100 score does not match original report")
    deltas = {key:result["base"][key] - old for key,old in V3_BASE.items()}
    return {
        "status": "HOLD_CROSS_RUN_BASELINE_DRIFT",
        "historical_v3_run": RUN_V3,
        "latest_v8_run": RUN_V8,
        "historical_v3_base_from_original_summary_log": V3_BASE,
        "latest_v8_result_recomputed_from_original_20_shards": result,
        "old_to_new_base_delta": deltas,
        "same_question_selection_sha256": EXPECTED_SELECTION_SHA,
        "same_pinned_base_model_evaluator_and_llama": True,
        "historical_per_task_v3_artifacts_available": False,
        "unresolved_confounders": [
            "Live external search content and public GAIA answer exposure",
            "Unpinned transitive Python dependencies and evolving runtime services",
            "Independent stochastic tool/model trajectories despite temperature=0",
        ],
        "promotion_allowed": False,
        "note": "Difference is real; no post-hoc causal attribution or decontaminated score claim.",
    }

def main():
    out = audit(Path("evidence"))
    Path("audit").mkdir(exist_ok=True)
    Path("audit/gaia-base9-to-base20.json").write_text(json.dumps(out, indent=2,sort_keys=True)+"\n")
    print("GAIA_BASELINE_DRIFT_VERIFIED", json.dumps(out["old_to_new_base_delta"]))

if __name__ == "__main__":
    main()
