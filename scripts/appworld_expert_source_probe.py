"""CPU-only AppWorld TRAIN expert-source feasibility; no model or training.

Run trusted upstream AppWorld TRAIN solutions in fresh isolated worlds.
ONLY aggregate counts leave the runner. Protected program, task prompts,
API arguments, app states and evaluator details never enter stdout/artifacts.

Official expert program execution is NOT a verified Qwen trajectory or
prefix replay. This is an evidence acquisition preflight, not SFT data.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import tempfile

SEED = "appworld-train-source-feasibility-fixed-prelabel-20261122"
UPSTREAM_SHA = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"
N_SCENES = 12


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def select_train_tasks(task_ids: list[str], *, count: int = N_SCENES) -> list[str]:
    if len(task_ids) != len(set(task_ids)) or not task_ids:
        raise ValueError("duplicate_or_empty_task_roster")
    grouped: dict[str, list[str]] = {}
    for task_id in task_ids:
        if not task_id or "_" not in task_id:
            raise ValueError("malformed_task_id")
        grouped.setdefault(task_id.split("_", 1)[0], []).append(task_id)
    if len(grouped) < count or count < 1:
        raise ValueError("insufficient_distinct_scenes")
    scenes = sorted(grouped, key=lambda s: digest(SEED + ":scene:" + s))[:count]
    return [min(grouped[s], key=lambda t: digest(SEED + ":task:" + t)) for s in scenes]


def safe_summary(rows: list[dict], *, source_sha256: str, sample_sha256: str) -> dict:
    if not rows or any(not re.fullmatch(r"[0-9a-f]{64}", s) for s in (source_sha256, sample_sha256)):
        raise ValueError("invalid_source_evidence")
    for row in rows:
        if type(row.get("official_expert_success")) is not bool:
            raise ValueError("missing_boolean_official_success")
        count = row.get("publisher_api_call_count")
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("invalid_publisher_api_count")
    successful = sum(int(row["official_expert_success"]) for row in rows)
    observed_depth = sum(row.get("publisher_api_call_count") is not None for row in rows)
    depth_ge7 = sum(
        (row.get("publisher_api_call_count") or 0) >= 7 for row in rows
    )
    return {
        "protocol": "APPWORLD-EXPERT-TRAIN-SOURCE-FEASIBILITY-CPU-V1",
        "upstream_sdk_commit": UPSTREAM_SHA,
        "source_roster_sha256": source_sha256,
        "fixed_prelabel_selection_sha256": sample_sha256,
        "n_distinct_train_task_families": len(rows),
        "official_expert_terminal_success": successful,
        "publisher_api_depth_observed": observed_depth,
        "publisher_declared_api_depth_at_least_7": depth_ge7,
        "verified_native_agent_long_horizon_success": 0,
        "native_prefix_state_replays": 0,
        "native_stop_labels": {"STOP": 0, "CONTINUE": 0, "UNKNOWN": len(rows)},
        "research_decision": "SOURCE_EXPERT_FEASIBLE_NOT_NATIVE_AGENT_SUPERVISION" if successful else "NO_OFFICIAL_EXPERT_SUCCESS_REPLAYED",
        "no_qwen_or_training": True,
        "no_raw_task_solutions_states_credentials_or_trace_export": True,
    }


def probe() -> dict:
    from appworld import AppWorld, load_task_ids

    roster = load_task_ids("train")
    chosen = select_train_tasks(roster)
    roster_sha = digest("\n".join(sorted(roster)))
    selected_sha = digest("\n".join(chosen))
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="private-appworld-expert-") as base:
        for i, task_id in enumerate(chosen):
            stdout, stderr = io.StringIO(), io.StringIO()
            success = False
            declared_api_count = None
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with AppWorld(
                        task_id=task_id,
                        experiment_name=f"private-expert-source-probe-{i}",
                        ground_truth_mode="full",
                        timeout_seconds=180,
                    ) as world:
                        gt = world.task.ground_truth
                        declared = getattr(gt, "api_calls", None)
                        if isinstance(declared, (list, tuple)):
                            declared_api_count = len(declared)
                        code = gt.compiled_solution_code
                        # Never persist or print official solution/feedback.
                        world.execute(code + "\nsolution(apis, requester)")
                        tracker = world.evaluate(suppress_errors=True)
                        success = bool(tracker.success)
            except Exception:
                # AppWorld errors may include credentials or task text.
                success = False
            finally:
                stdout.close()
                stderr.close()
            rows.append({
                "official_expert_success": success,
                "publisher_api_call_count": declared_api_count,
            })
    return safe_summary(rows, source_sha256=roster_sha, sample_sha256=selected_sha)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("immutable_probe_output")
    result = probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("OFFICIAL_EXPERT_SOURCE_FEASIBILITY_COUNTS", json.dumps({
        "tasks": result["n_distinct_train_task_families"],
        "official_success": result["official_expert_terminal_success"],
        "declared_api_depth_ge7": result["publisher_declared_api_depth_at_least_7"],
        "native_supervision_labels": result["native_stop_labels"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
