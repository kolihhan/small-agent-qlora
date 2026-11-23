"""Pure metadata-only GAIA100 stability aggregation; never reads reference answers."""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

EXPECTED_ARMS = ("base", "v8step70")
REPEATS = (2, 3)


def load_repeat(root: Path, repeat: int, original: dict) -> dict:
    old_ids = [r["task_hash"] for r in original["per_task"]]
    arms = {}
    for arm in EXPECTED_ARMS:
        cases = []
        for shard in range(20):
            path = root / f"repeat-{repeat}-{arm}-{shard}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["schema"] == "gaia-fourway-same-task/v1"
            assert data["arm"] == arm and data["shard"] == shard
            assert data["full_ids_sha256"] == original["full_ids_sha256"]
            assert data["dataset_revision"] == original["dataset_revision"]
            assert data["runtime"] == {
                "steps": 12, "temperature": 0, "reasoning_budget": 0,
                "context_tokens": 16384, "max_new_tokens": 512,
                "tools": ["search", "read", "inspect", "python"],
            }, "protocol drift"
            assert len(data["cases"]) == 5
            cases += data["cases"]
        assert len(cases) == 100
        assert [r["task_hash"] for r in cases] == old_ids, "task identity/order drift"
        arms[arm] = cases

    base, v8 = arms["base"], arms["v8step70"]
    wins = [b["task_hash"] for b, v in zip(base, v8) if not b["correct"] and v["correct"]]
    losses = [b["task_hash"] for b, v in zip(base, v8) if b["correct"] and not v["correct"]]
    return {
        "repeat": repeat,
        "base_correct": sum(x["correct"] for x in base),
        "v8step70_correct": sum(x["correct"] for x in v8),
        "base_completed": sum(x["completed"] for x in base),
        "v8step70_completed": sum(x["completed"] for x in v8),
        "base_invalid_final": sum(x["stop_reason"] == "invalid_final" for x in base),
        "v8step70_invalid_final": sum(x["stop_reason"] == "invalid_final" for x in v8),
        "base_tool_errors": sum(x["tool_errors"] for x in base),
        "v8step70_tool_errors": sum(x["tool_errors"] for x in v8),
        "v8_wins": wins, "v8_losses": losses,
        "delta_correct": len(wins) - len(losses),
        "cases": [
            {"task_hash": b["task_hash"], "base_correct": b["correct"],
             "v8step70_correct": v["correct"],
             "base_stop": b["stop_reason"], "v8step70_stop": v["stop_reason"]}
            for b, v in zip(base, v8)
        ],
    }


def main() -> None:
    original = json.loads(Path("original/gaia-same-task-summary.json").read_text(encoding="utf-8"))
    assert original["n"] == 100
    assert set(original["arm_metrics"]) == {"base", "v7", "v8step70"}
    assert len(original["per_task"]) == 100
    o = original["arm_metrics"]
    first = {
        "repeat": 1,
        "base_correct": o["base"]["correct"],
        "v8step70_correct": o["v8step70"]["correct"],
        "base_completed": o["base"]["completed"],
        "v8step70_completed": o["v8step70"]["completed"],
        "base_invalid_final": o["base"]["invalid_final"],
        "v8step70_invalid_final": o["v8step70"]["invalid_final"],
        "base_tool_errors": o["base"]["tool_errors"],
        "v8step70_tool_errors": o["v8step70"]["tool_errors"],
        "v8_wins": original["paired_vs_base"]["v8step70"]["wins"],
        "v8_losses": original["paired_vs_base"]["v8step70"]["losses"],
        "delta_correct": o["v8step70"]["correct"] - o["base"]["correct"],
    }
    results = [first, *(load_repeat(Path("repeats"), t, original) for t in REPEATS)]
    report = {
        "status": "COMPLETE_3_MATCHED_REPEATS_SEEN_GAIA100",
        "scope": "same frozen GAIA100, three independent executions; NOT unseen held-out",
        "model": "Qwen3.5-9B Q4_K_M, Base versus physically verified QLoRA v8 Step70",
        "original_run": 45144537782,
        "data_revision": original["dataset_revision"],
        "cohort_hash": original["full_ids_sha256"],
        "repeats": results,
        "mean_base_correct": mean(x["base_correct"] for x in results),
        "mean_v8step70_correct": mean(x["v8step70_correct"] for x in results),
        "mean_paired_delta": mean(x["delta_correct"] for x in results),
        "all_positive": all(x["delta_correct"] > 0 for x in results),
        "no_new_training": True,
        "no_promotion_without_independent_transfer": True,
    }
    Path("gaia-v8-stability-3runs.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# GAIA100 Qwen3.5-9B v8 Step70 stability: 3 matched runs",
        "",
        "**Previously seen GAIA100 DEV, not held-out.** Original first run is preserved.",
        "",
        "| Run | Base correct | v8 Step70 correct | Paired delta | v8 wins | v8 losses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(f"| {r['repeat']} | {r['base_correct']}/100 | {r['v8step70_correct']}/100 | "
                     f"{r['delta_correct']:+d} | {len(r['v8_wins'])} | {len(r['v8_losses'])} |")
    lines += ["", f"Mean delta: {report['mean_paired_delta']:+.2f} / 100.",
              f"All three positive: {report['all_positive']}.", "",
              "This proves repeatability on one frozen previously seen cohort only.",
              "No hidden GAIA result or independent source-disjoint transfer is claimed.",
              ""]
    Path("gaia-v8-stability-3runs.md").write_text("\n".join(lines), encoding="utf-8")
    print("GAIA_V8_STABILITY", json.dumps({
        "counts": [(r["base_correct"], r["v8step70_correct"]) for r in results],
        "mean_delta": report["mean_paired_delta"], "all_positive": report["all_positive"],
    }), flush=True)


if __name__ == "__main__":
    main()
