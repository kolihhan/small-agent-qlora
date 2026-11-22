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
import subprocess
import sys

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



SAFE_PREFIX = "APPWORLD_SAFE_AGGREGATE="
SAFE_FIELDS = frozenset({
    "protocol", "upstream_sdk_commit", "source_roster_sha256",
    "fixed_prelabel_selection_sha256", "n_distinct_train_task_families",
    "official_expert_terminal_success", "publisher_api_depth_observed",
    "publisher_declared_api_depth_at_least_7",
    "verified_native_agent_long_horizon_success", "native_prefix_state_replays",
    "native_stop_labels", "research_decision", "no_qwen_or_training",
    "no_raw_task_solutions_states_credentials_or_trace_export",
})


def parse_safe_worker_output(stdout: str) -> dict:
    """Parse only one explicitly marked, strictly allowlisted worker result."""
    rows = [line[len(SAFE_PREFIX):] for line in stdout.splitlines()
            if line.startswith(SAFE_PREFIX)]
    if len(rows) != 1:
        raise ValueError("worker_safe_report_missing_or_duplicate")
    try:
        d = json.loads(rows[0])
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid_worker_json") from exc
    if (not isinstance(d, dict) or set(d) != SAFE_FIELDS
            or d.get("protocol") != "APPWORLD-EXPERT-TRAIN-SOURCE-FEASIBILITY-CPU-V1"
            or d.get("upstream_sdk_commit") != UPSTREAM_SHA
            or d.get("no_qwen_or_training") is not True
            or d.get("no_raw_task_solutions_states_credentials_or_trace_export") is not True):
        raise ValueError("unsafe_worker_report_schema")
    if (type(d.get("n_distinct_train_task_families")) is not int
            or d["n_distinct_train_task_families"] < 1
            or d.get("native_stop_labels") != {
                "STOP": 0, "CONTINUE": 0, "UNKNOWN": d["n_distinct_train_task_families"]
            }):
        raise ValueError("unexpected_native_supervision_claim")
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        # The official SDK's sandbox patches pathlib/shutil in this process.
        # No I/O is attempted after the AppWorld work: only safe stdout.
        print(SAFE_PREFIX + json.dumps(probe(), sort_keys=True))
        return
    if args.out.exists():
        raise FileExistsError("immutable_probe_output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.run(
        [sys.executable, "-m", "scripts.appworld_expert_source_probe",
         "--worker", "--out", str(args.out)],
        capture_output=True, text=True, timeout=900, check=False,
    )
    # Never log raw child stdout/stderr; SDK errors may contain protected data.
    if child.returncode:
        raise SystemExit("private_appworld_worker_failed_no_sensitive_logs")
    result = parse_safe_worker_output(child.stdout)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("OFFICIAL_EXPERT_SOURCE_FEASIBILITY_COUNTS", json.dumps({
        "tasks": result["n_distinct_train_task_families"],
        "official_success": result["official_expert_terminal_success"],
        "declared_api_depth_ge7": result["publisher_declared_api_depth_at_least_7"],
        "native_supervision_labels": result["native_stop_labels"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
