"""AppWorld TRAIN original-vs-one-public-allowed-app-list exact-prefix fork.

No oracle, no task-specific answers, no new model, no adapter. The only treatment
is the same public task_apps list available from appworld_docs(app omitted), now
appended to the FIRST APP_DENIED appworld_docs tool error. Only fork when the
student has NOT already seen that list and can take a subsequent model step.
Actual outputs contain no protected prompts, API values or task identifiers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool, AppWorldDocsTool
from gaia_small_agent.tools.base import ToolResult
from scripts.appworld_train_first_feedback_fork import (
    ExactPrefixModel, RecordingModel, MAX_API_CALLS, MAX_STEPS,
    MODEL_GGUF_SHA, UPSTREAM_APPWORLD_REV, digest, new_model,
    prompt_for, task_roster, wall_ns, first_observed_error,
)

PROTOCOL = "APPWORLD-QWEN35-TRAIN-FIRST-DENIED-APP-LIST-PREFIX-V1"


def public_allowed_apps(world):
    return sorted(set(world.task.allowed_apps) - {"admin", "api_docs"})


def safe_hint(world):
    # Identical keys/values to appworld_docs when the model omits app.
    return "Public task apps: " + json.dumps({
        "task_apps": public_allowed_apps(world),
        "supervisor_completion": "supervisor.complete_task",
    }, sort_keys=True)


class OneShotAllowedAppsDocsTool(AppWorldDocsTool):
    def __init__(self, world, target_call_index):
        super().__init__(world)
        self.target_call_index = target_call_index
        self.calls = 0
        self.hint_delivered = False

    def run(self, arguments, workspace):
        self.calls += 1
        result = super().run(arguments, workspace)
        if self.calls == self.target_call_index and result.error_code == "APP_DENIED":
            self.hint_delivered = True
            return ToolResult(False, result.content + " " + safe_hint(self.world), "APP_DENIED")
        return result


def prior_public_app_list_visible(trace, error_step, error_id):
    # Trace order is the actual in-step call execution order; no raw content
    # will ever be written to the public evidence record.
    for event in trace:
        if event.kind == "tool_result" and event.data.get("id") == error_id and event.step == error_step:
            break
        if (event.kind == "tool_result" and event.data.get("ok")
                and event.data.get("name") == "appworld_docs"
                and '"task_apps"' in event.data.get("content", "")):
            return True
    return False


def first_denied_docs_error(trace):
    first = first_observed_error(trace)
    if not first or first["kind"] != "APP_DENIED" or first["name"] != "appworld_docs":
        return None, "FIRST_OBSTRUCTION_NOT_DOCS_APP_DENIED"
    index = 0
    last_call = None
    for ev in trace:
        if ev.kind == "tool_call" and ev.data.get("name") == "appworld_docs":
            index += 1
            last_call = ev.data.get("id")
        if (ev.kind == "tool_result" and ev.step == first["step"]
                and ev.data.get("id") == last_call
                and ev.data.get("name") == "appworld_docs"
                and ev.data.get("error_code") == "APP_DENIED"):
            if prior_public_app_list_visible(trace, ev.step, ev.data.get("id")):
                return None, "PUBLIC_APP_LIST_ALREADY_SEEN_BEFORE_DENIAL"
            return {"step": ev.step, "docs_call_index": index}, "ELIGIBLE_FIRST_DENIED_APP"
    return None, "DENIAL_EVENT_NOT_IDENTIFIED"


def post_error_correctly_scoped_app_call(trace, after_step, allowed):
    for ev in trace:
        if ev.kind != "tool_call" or ev.step <= after_step:
            continue
        if ev.data.get("name") not in ("appworld_docs", "appworld_call"):
            continue
        app = ev.data.get("arguments", {}).get("app")
        if app is None:
            continue
        return app in set(allowed) or app == "supervisor"
    return False


def evaluate_one(index, output_dir):
    from appworld import AppWorld
    roster = task_roster()
    task_id = roster[index]
    task_hash = sha256(task_id.encode("utf-8")).hexdigest()
    begun = wall_ns()
    with AppWorld(task_id=task_id, experiment_name=f"qwen35-app-list-original-{index}",
                  ground_truth_mode="full", timeout_seconds=180) as world:
        allowed = public_allowed_apps(world)
        recorder = RecordingModel(new_model())
        api = AppWorldCallTool(world, max_calls=MAX_API_CALLS,
                               report_http_failures=True)
        episode = AgentRuntime(recorder, [AppWorldDocsTool(world), api],
                               max_steps=MAX_STEPS).run(prompt_for(world), output_dir / "original")
        score = world.evaluate(suppress_errors=True)
        first = first_observed_error(episode.trace)
        target, status = first_denied_docs_error(episode.trace)
        allowed_hash = digest(allowed)
        original_scoped = post_error_correctly_scoped_app_call(
            episode.trace, first["step"], allowed) if first else False
    record = {
        "protocol": PROTOCOL,
        "appworld_commit": UPSTREAM_APPWORLD_REV,
        "qwen_gguf_sha256": MODEL_GGUF_SHA,
        "roster_hash": digest(roster), "task_hash": task_hash,
        "public_app_list_sha256": allowed_hash,
        "shard": index, "max_steps": MAX_STEPS, "max_api_calls": MAX_API_CALLS,
        "original": {
            "official_task_success": bool(score.success),
            "nested_state_tests": [int(score.pass_count), int(score.num_tests)],
            "model_calls": len(recorder.records),
            "tool_calls": episode.metrics.tool_calls,
            "api_calls": api.calls, "http_errors": api.http_errors,
            "stop_reason": episode.stop_reason,
            "first_observed_tool_error": first,
            "first_error_followed_by_correctly_scoped_app_call": original_scoped,
            "elapsed_raw_wall_s": round((wall_ns()-begun)/1e9, 2),
        },
        "fork_status": status,
        "same_state_fork_scored": False,
        "model_never_saw_gold": True,
        "no_adapter_or_training": True, "GAIA_not_run": True,
        "do_not_log_task_or_raw_api": True,
    }
    if target is None:
        return record
    step = target["step"]
    if not isinstance(step, int) or step >= len(recorder.records):
        record["fork_status"] = "NO_NEXT_MODEL_STEP_AFTER_DENIAL"
        return record

    # A new simulator gets the same task, tool inputs and frozen Qwen turns.
    # ExactPrefixModel aborts if ANY prior model-visible input differs and
    # rejects anything except exactly one appended public tool observation.
    fork_begun = wall_ns()
    with AppWorld(task_id=task_id, experiment_name=f"qwen35-app-list-fork-{index}",
                  ground_truth_mode="full", timeout_seconds=180) as world:
        model = ExactPrefixModel(new_model(), recorder.records,
                                 prefix_steps=step, hint=safe_hint(world))
        doc = OneShotAllowedAppsDocsTool(world, target["docs_call_index"])
        api = AppWorldCallTool(world, max_calls=MAX_API_CALLS,
                               report_http_failures=True)
        fork = AgentRuntime(model, [doc, api], max_steps=MAX_STEPS).run(
            prompt_for(world), output_dir / "public-app-list-only")
        if not model.one_observation_delta_verified or not doc.hint_delivered:
            raise RuntimeError("no exact one-public-app-list feedback fork")
        if model.verified_original_prefix_calls != step:
            raise RuntimeError("prior Qwen history was not identical")
        score2 = world.evaluate(suppress_errors=True)
        treatment_scoped = post_error_correctly_scoped_app_call(
            fork.trace, step, public_allowed_apps(world))
    record.update({
        "fork_status": "EXACT_PREFIX_ONE_PUBLIC_APP_LIST_OBSERVATION_VERIFIED",
        "same_state_fork_scored": True,
        "treatment": {
            "official_task_success": bool(score2.success),
            "nested_state_tests": [int(score2.pass_count), int(score2.num_tests)],
            "new_autonomous_model_calls": model.new_model_calls,
            "verified_original_prefix_model_calls": model.verified_original_prefix_calls,
            "first_postfork_input_exactly_one_public_list_delta": True,
            "first_error_followed_by_correctly_scoped_app_call": treatment_scoped,
            "tool_calls": fork.metrics.tool_calls,
            "api_calls": api.calls, "http_errors": api.http_errors,
            "stop_reason": fork.stop_reason,
            "elapsed_raw_wall_s": round((wall_ns()-fork_begun)/1e9, 2),
        },
        "first_wrong_app_newly_corrected": not original_scoped and treatment_scoped,
        "task_rescue": not bool(record["original"]["official_task_success"]) and bool(score2.success),
        "task_regression": bool(record["original"]["official_task_success"]) and not bool(score2.success),
    })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, choices=range(8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse to overwrite physical evidence")
    try:
        row = evaluate_one(args.task_index, args.out.parent / f"private-{args.task_index}")
    except Exception as exc:
        # Never leak private AppWorld task or API contents in public Actions logs.
        print("TRAIN_FAIL_CLOSED_EXCEPTION_TYPE", type(exc).__name__, flush=True)
        raise SystemExit(1) from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print("QWEN35_TRAIN_APP_LIST_PREFIX", json.dumps({
        "shard": row["shard"],
        "fork_status": row["fork_status"],
        "original_official_success": row["original"]["official_task_success"],
        "fork_scored": row["same_state_fork_scored"],
        "treatment_official_success": row.get("treatment", {}).get("official_task_success"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
