"""Autonomous Qwen3.5-9B Base vs physically verified v8 adapter on 24 full MuSiQue workspaces.

Source-frozen TRAIN-derived holdout, not GAIA and not a training claim.
Do not run without independent v8 checkpoint and SHA verification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
from hashlib import sha256
import json
from pathlib import Path
import time

from gaia_small_agent.training.trajectories import _materialize_files
from gaia_small_agent.benchmark.scoring import gaia_score
from scripts.qlora_v8_holdout_paired import MODEL, validate_inputs


def paired_results(base: list[dict], candidate: list[dict]) -> dict:
    """Compare task-level end-to-end correctness, not oracle-prefix next actions."""
    if (len(base) != len(candidate)
        or len({x["task_id"] for x in base}) != len(base)
        or {x["task_id"] for x in base} != {x["task_id"] for x in candidate}):
        raise ValueError("paired task coverage drift")
    b = {x["task_id"]: x for x in base}
    c = {x["task_id"]: x for x in candidate}
    total = len(b)
    counts = Counter()
    hop = defaultdict(Counter)
    examples = []
    for ident in sorted(b):
        old, new = b[ident], c[ident]
        if old["hop_count"] != new["hop_count"]:
            raise ValueError("hop/source protocol drift")
        gain = int(not old["correct"] and new["correct"])
        loss = int(old["correct"] and not new["correct"])
        for bucket in (counts, hop[str(old["hop_count"])]):
            bucket.update({
                "total": 1, "base_correct": int(old["correct"]),
                "v8_correct": int(new["correct"]), "rescues": gain, "regressions": loss,
                "base_completed": int(old["completed"]), "v8_completed": int(new["completed"]),
                "base_calls": old["tool_calls"], "v8_calls": new["tool_calls"],
                "base_duplicates": old["duplicate_calls_blocked"],
                "v8_duplicates": new["duplicate_calls_blocked"],
            })
        examples.append({
            "task_id": ident, "hop_count": old["hop_count"],
            "base_correct": old["correct"], "v8_correct": new["correct"],
            "base_stop": old["stop_reason"], "v8_stop": new["stop_reason"],
        })
    if counts["v8_correct"] - counts["base_correct"] != counts["rescues"] - counts["regressions"]:
        raise AssertionError("paired transition accounting drift")
    return {
        **dict(counts), "by_hop": {k: dict(v) for k, v in sorted(hop.items())},
        "cases": examples,
        "metric": "real autonomous exact task success, same 20-passage workspaces, max 12 agent steps",
        "promotion_allowed": False,
        "limitation": "TRAIN-origin source-family holdout, only 24 MuSiQue tasks; independent GAIA/OOD evaluation remains necessary",
    }


def _evaluate(model, tasks: list[dict], root: Path) -> list[dict]:
    from gaia_small_agent.agent.loop import AgentRuntime
    from gaia_small_agent.tools import default_tools
    output = []
    for row in tasks:
        task = row["eval_task"]
        workspace = root / row["task_id"]
        _materialize_files(task, workspace)
        runtime = AgentRuntime(model, default_tools(), max_steps=12)
        result = runtime.run(task["question"], workspace)
        output.append({
            "task_id": row["task_id"], "hop_count": row["hop_count"],
            "completed": bool(result.completed),
            "correct": bool(result.completed and gaia_score(result.answer, task["expected_answer"])),
            "stop_reason": str(result.stop_reason),
            "steps": result.metrics.steps,
            "tool_calls": result.metrics.tool_calls,
            "tool_errors": result.metrics.tool_errors,
            "duplicate_calls_blocked": result.metrics.duplicate_calls_blocked,
        })
    return output


def run(tasks_path: Path, source_report: Path, holdout: Path, audit: Path, adapter: Path) -> dict:
    # Reuse the previously tested 100-step checkpoint/physical weight verification.
    proof = validate_inputs(holdout, audit, adapter)
    rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = json.loads(source_report.read_text(encoding="utf-8"))
    if (source.get("protocol") != "v8-musique-real-20passage-autonomous-probe-v1"
        or source.get("input_validation_sha256") != proof["audit"]["validation_sha256"]
        or source.get("eligible_mu_parents") != 24
        or len(rows) != 24
        or sha256("\n".join(x["task_id"] for x in rows).encode()).hexdigest() != source.get("task_manifest_sha256")
        or len({x["task_id"] for x in rows}) != 24
        or any(len(x["eval_task"]["files"]) != 21 for x in rows)):
        raise ValueError("source-complete holdout identity/coverage drift")

    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 14 * 2**30:
        raise RuntimeError("v8 9B autonomous paired evaluation requires real >=14GiB CUDA")
    from gaia_small_agent.model.transformers_qwen import TransformersQwenModel

    results = {}
    elapsed = {}
    for label, path in (("base", None), ("v8", str(adapter))):
        start = time.monotonic()
        model = TransformersQwenModel(
            model_name=MODEL, quantize_4bit=True, adapter=path,
            enable_thinking=False, max_new_tokens=512, load_compute_dtype=True,
        )
        try:
            results[label] = _evaluate(model, rows, Path("workspaces") / label)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        elapsed[label] = round(time.monotonic() - start, 1)

    return {
        "protocol": "v8-musique24-independent-autonomous-paired-v1",
        "model": MODEL, "max_steps": 12, "max_new_tokens": 512,
        "source_tasks_sha256": sha256(tasks_path.read_bytes()).hexdigest(),
        "source_report_sha256": sha256(source_report.read_bytes()).hexdigest(),
        "holdout_sha256": proof["audit"]["validation_sha256"],
        "checkpoint_sha256": proof["checkpoint"]["checkpoint_sha256"],
        "source_sha": proof["checkpoint"]["source_sha"],
        "latency_s": elapsed,
        "paired": paired_results(results["base"], results["v8"]),
        "promotion_allowed": False,
    }


def main():
    p = argparse.ArgumentParser()
    for key in ("tasks", "source-report", "holdout", "audit", "adapter", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwrite of evaluation artifact")
    output = run(a.tasks, a.source_report, a.holdout, a.audit, a.adapter)
    a.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("V8_INDEPENDENT_AUTONOMOUS_PAIRED", json.dumps({k: v for k, v in output["paired"].items() if k != "cases"}, sort_keys=True))


if __name__ == "__main__":
    main()
