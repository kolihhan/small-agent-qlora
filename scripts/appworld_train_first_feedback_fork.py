"""Exact-prefix one-field AppWorld TRAIN feedback fork with original Qwen3.5-9B.

One real original student episode runs to completion. The first observed tool
failure must be an HTTP 422 error with a publicly documented parameter hint.
A new AppWorld simulator replays original model tool decisions up to that
failure, checking each model-visible prefix byte-for-byte before continuing
with only ONE appended public-field hint. All post-fork decisions are made
autonomously by the same Qwen model. Official state evaluation occurs only
after each student finishes. Protected prompt/actions never leave the runner.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool, AppWorldDocsTool

UPSTREAM_APPWORLD_REV = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"
MODEL_GGUF_SHA = "8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
MAX_STEPS = 20
MAX_API_CALLS = 48
COHORT_SEED = "qwen35-9b-train-first-obstruction-prefix-lock-v1"


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             default=str).encode("utf-8")).hexdigest()


def wall_ns():
    # AppWorld patches perf_counter; use raw monotonic clock.
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class RecordingModel:
    def __init__(self, delegate):
        self.delegate = delegate
        self.records = []

    def _run(self, kind, messages, tools):
        func = getattr(self.delegate, kind, None)
        if not callable(func):
            if kind != "complete_final":
                raise RuntimeError("missing model entrypoint")
            func = self.delegate.complete
            tools = []
        turn = func(messages, tools)
        self.records.append({"kind": kind, "messages": deepcopy(messages),
                             "turn": deepcopy(turn)})
        return turn

    def complete(self, messages, tools):
        return self._run("complete", messages, tools)

    def complete_final(self, messages, tools):
        return self._run("complete_final", messages, tools)


class ExactPrefixModel:
    def __init__(self, delegate, original_records, prefix_steps, hint):
        self.delegate = delegate
        self.records = original_records
        self.prefix_steps = prefix_steps
        self.hint = hint
        self.calls = 0
        self.new_model_calls = 0
        self.verified_original_prefix_calls = 0
        self.one_observation_delta_verified = False
        if prefix_steps <= 0 or prefix_steps >= len(original_records):
            raise ValueError("same-step fork requires next autonomous decision")

    def _check_one_delta(self, messages, original):
        if len(messages) != len(original):
            raise ValueError("tool prefix length diverged")
        different = [(i, a, b) for i, (a, b) in
                     enumerate(zip(original, messages)) if a != b]
        if len(different) != 1:
            raise ValueError("fork had more than one model-visible change")
        _, a, b = different[0]
        if (not isinstance(a, dict) or not isinstance(b, dict)
                or a.get("role") != "tool" or b.get("role") != "tool"):
            raise ValueError("only tool feedback may change")
        before = a.get("content")
        after = b.get("content")
        if (not isinstance(before, str) or not isinstance(after, str)
                or after != before + " " + self.hint):
            raise ValueError("counterfactual changed more than public 422 field")
        stripped = dict(b)
        stripped["content"] = before
        if stripped != a:
            raise ValueError("tool metadata or identity drift")
        self.one_observation_delta_verified = True

    def _run(self, kind, messages, tools):
        self.calls += 1
        index = self.calls - 1
        if self.calls <= self.prefix_steps:
            prior = self.records[index]
            if prior["kind"] != kind or digest(prior["messages"]) != digest(messages):
                raise ValueError("prior AppWorld state/observation is not matched")
            self.verified_original_prefix_calls += 1
            return deepcopy(prior["turn"])
        if self.calls == self.prefix_steps + 1:
            prior = self.records[index]
            if prior["kind"] != kind:
                raise ValueError("post-error first model-call mode changed")
            self._check_one_delta(messages, prior["messages"])
        func = getattr(self.delegate, kind, None)
        if not callable(func):
            if kind != "complete_final":
                raise RuntimeError("model method missing")
            func, tools = self.delegate.complete, []
        self.new_model_calls += 1
        return func(messages, tools)

    def complete(self, messages, tools):
        return self._run("complete", messages, tools)

    def complete_final(self, messages, tools):
        return self._run("complete_final", messages, tools)


class OneShotHintTool(AppWorldCallTool):
    def __init__(self, world, target_api_call):
        super().__init__(world, max_calls=MAX_API_CALLS,
                         report_http_failures=True)
        self.target_api_call = target_api_call
        self.hint_actually_delivered = False

    def run(self, arguments, workspace):
        target = self.calls + 1 == self.target_api_call
        self.public_422_field_hint = target
        try:
            result = super().run(arguments, workspace)
        finally:
            self.public_422_field_hint = False
        if target and result.error_code == "HTTP_ERROR" and (
                "Public API validation:" in result.content):
            self.hint_actually_delivered = True
        return result


def prompt_for(world):
    return (
        "You are operating in an isolated AppWorld simulation. "
        "Perform the following task by actually calling the available "
        "AppWorld APIs. Use appworld_docs to inspect available task apps "
        "and each API's required parameters; then use appworld_call "
        "to read and update the task state. Do not invent results. "
        "Call supervisor.complete_task when the requested changes are done. "
        "A final textual claim does not itself change app state.\n\n"
        "USER TASK:\n" + world.task.instruction
    )


def new_model():
    return OpenAICompatModel(
        model="qwen3.5-9b", base_url="http://127.0.0.1:8080",
        timeout_s=240, max_new_tokens=384)


def task_roster():
    from appworld import load_task_ids
    ids = load_task_ids("train")
    ranked = sorted(ids, key=lambda tid: sha256(
        (COHORT_SEED + ":" + tid).encode("utf-8")).digest())
    chosen, scenes = [], set()
    for task_id in ranked:
        scene = task_id.split("_", 1)[0]
        if scene not in scenes:
            chosen.append(task_id)
            scenes.add(scene)
        if len(chosen) >= 8:
            break
    if len(chosen) != 8:
        raise RuntimeError("expected eight distinct source TRAIN scenarios")
    return chosen


def first_observed_error(trace):
    for event in trace:
        if event.kind == "tool_result" and not event.data.get("ok", False):
            return {"step": event.step,
                    "kind": event.data.get("error_code"),
                    "name": event.data.get("name")}
    return None


def successful_api_after(trace, step):
    calls = {}
    for event in trace:
        if event.kind == "tool_call":
            calls[event.data["id"]] = event.data
        if event.kind == "tool_result" and event.step > step and (
                event.data.get("ok") and event.data.get("name") == "appworld_call"):
            return True
    return False


def evaluate_one(index, output_dir):
    from appworld import AppWorld
    roster = task_roster()
    task_id = roster[index]
    source_hash = digest(roster)
    task_hash = sha256(task_id.encode("utf-8")).hexdigest()
    begun = wall_ns()
    name = "qwen35-train-original-obstruction-" + str(index)
    with AppWorld(task_id=task_id, experiment_name=name,
                  ground_truth_mode="full", timeout_seconds=180) as world:
        recorder = RecordingModel(new_model())
        api = AppWorldCallTool(world, max_calls=MAX_API_CALLS,
                               report_http_failures=True,
                               capture_422_field_hint=True)
        api.model_step_counter = lambda: len(recorder.records)
        student = AgentRuntime(
            recorder, [AppWorldDocsTool(world), api],
            max_steps=MAX_STEPS).run(prompt_for(world), output_dir / "original")
        verdict = world.evaluate(suppress_errors=True)
        original_score = bool(verdict.success)
        original_nested = [int(verdict.pass_count), int(verdict.num_tests)]
    original_elapsed = round((wall_ns() - begun) / 1e9, 2)
    first = first_observed_error(student.trace)
    record = {
        "protocol": "APPWORLD-QWEN35-TRAIN-FIRST-HTTP-FIELD-FIXED-PREFIX-FORK-V1",
        "appworld_commit": UPSTREAM_APPWORLD_REV,
        "qwen_gguf_sha256": MODEL_GGUF_SHA,
        "task_hash": task_hash,
        "roster_hash": source_hash,
        "shard": index, "max_steps": MAX_STEPS,
        "max_api_calls": MAX_API_CALLS,
        "original": {
            "official_task_success": original_score,
            "nested_state_tests": original_nested,
            "model_calls": len(recorder.records),
            "tool_calls": student.metrics.tool_calls,
            "api_calls": api.calls,
            "http_errors": api.http_errors,
            "stop_reason": student.stop_reason,
            "elapsed_raw_wall_s": original_elapsed,
            "first_observed_tool_error": first,
        },
        "eligible_documented_422_count": len(api.public_422_candidates),
        "model_never_saw_gold": True,
        "do_not_log_task_or_raw_api": True,
        "same_state_fork_scored": False,
        "no_adapter_or_training": True,
        "GAIA_not_run": True,
    }
    if not api.public_422_candidates:
        record["fork_status"] = "NO_PUBLIC_DOCUMENTED_FIELD_422_OBSERVED"
        return record

    target = api.public_422_candidates[0]
    step = target.get("model_step")
    record["first_eligible_422"] = {
        "step": step, "http_status": target["http_status"],
        "api_call_index": target["api_call_index"],
        "visible_baseline_template": "HTTP 422; check API documentation and parameters",
        "only_public_documented_field_hint": target["hint"],
    }
    if not first or first["step"] != step or first["kind"] != "HTTP_ERROR":
        record["fork_status"] = "422_NOT_FIRST_OBSERVED_TOOL_OBSTRUCTION"
        return record
    if not isinstance(step, int) or step >= len(recorder.records):
        record["fork_status"] = "NO_NEXT_MODEL_STEP_AFTER_ERROR"
        return record

    fork_begun = wall_ns()
    with AppWorld(task_id=task_id,
                  experiment_name="qwen35-train-counterfactual-obstruction-" + str(index),
                  ground_truth_mode="full", timeout_seconds=180) as world:
        model = ExactPrefixModel(new_model(), recorder.records,
                                 step, target["hint"])
        api = OneShotHintTool(world, target["api_call_index"])
        student_fork = AgentRuntime(
            model, [AppWorldDocsTool(world), api],
            max_steps=MAX_STEPS).run(
                prompt_for(world), output_dir / "one_field_counterfactual")
        if not model.one_observation_delta_verified or not api.hint_actually_delivered:
            raise RuntimeError("one documented-field hint was not precisely delivered")
        if model.verified_original_prefix_calls != step:
            raise RuntimeError("original model/tool history not identical before failure")
        verdict = world.evaluate(suppress_errors=True)
        treat_score = bool(verdict.success)
        treat_nested = [int(verdict.pass_count), int(verdict.num_tests)]
    record["same_state_fork_scored"] = True
    record["fork_status"] = "EXACT_PREFIX_ONE_OBSERVATION_DELTA_PHYSICALLY_VERIFIED"
    record["treatment"] = {
        "official_task_success": treat_score,
        "nested_state_tests": treat_nested,
        "verified_original_prefix_model_calls": model.verified_original_prefix_calls,
        "new_autonomous_model_calls": model.new_model_calls,
        "all_model_calls_in_fork": model.calls,
        "first_postfork_input_exactly_one_public_field_delta": True,
        "http_errors": api.http_errors,
        "tool_calls": student_fork.metrics.tool_calls,
        "stop_reason": student_fork.stop_reason,
        "elapsed_raw_wall_s": round((wall_ns() - fork_begun) / 1e9, 2),
    }
    record["first_error_operationally_unblocked_original"] = successful_api_after(
        student.trace, step)
    record["first_error_operationally_unblocked_treatment"] = successful_api_after(
        student_fork.trace, step)
    record["first_error_newly_unblocked"] = (
        not record["first_error_operationally_unblocked_original"]
        and record["first_error_operationally_unblocked_treatment"])
    record["task_rescue"] = (not original_score and treat_score)
    record["task_regression"] = (original_score and not treat_score)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-index", type=int, choices=range(8), required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("never overwrite a physical evidence record")
    try:
        row = evaluate_one(a.task_index, a.out.parent / ("private-" + str(a.task_index)))
    except Exception as exc:
        # Never emit protected simulator/task/error text to public Actions logs.
        print("TRAIN_FAIL_CLOSED_EXCEPTION_TYPE", type(exc).__name__, flush=True)
        raise SystemExit(1) from None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print("QWEN35_TRAIN_ORIGINAL_AND_PREFIX_FORK", json.dumps({
        "shard": row["shard"], "status": row["fork_status"],
        "original_official_success": row["original"]["official_task_success"],
        "fork_scored": row["same_state_fork_scored"],
        "treatment_official_success":
            row.get("treatment", {}).get("official_task_success"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
