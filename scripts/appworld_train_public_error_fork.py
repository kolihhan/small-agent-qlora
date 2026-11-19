"""AppWorld TRAIN: replay the FIRST actually observed public HTTP error feedback.

Original frozen Qwen3.5-9B / AgentRuntime / 20-step tool policy. The control
runs autonomously. If it fails and an HTTP error hid exactly one documented
required public parameter, a second pristine world replays the EXACT original
model prefix through that tool call. Only its one feedback field is restored;
the same model then runs autonomously to termination. No reference action,
secret API, task/solution text or evaluator response enters model messages.

Only SHA/counters and generic feedback templates leave the runner. This
is a mechanism pilot, NOT QLoRA training, GAIA, or a population estimate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.appworld_tool import (
    AppWorldCallTool, AppWorldDocsTool, allowed_public_api,
)
from gaia_small_agent.tools.base import ToolResult
from scripts.replay_appworld_train_reference import choose_train_entries
from scripts.eval_appworld_qwen9b_dev57 import clock

PROTOCOL = "APPWORLD-TRAIN-QWEN9B-FIRST-PUBLIC-FIELD-EXACT-PREFIX-FORK-V1"
UPSTREAM_COMMIT = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"
GGUF_SHA = "8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
MAX_STEPS = 20
MAX_CALLS = 48
FEEDBACK_TEMPLATE = (
    "AppWorld API rejected this request (HTTP {status}). "
    "Check the API documentation and parameters."
)


def sha(value: Any) -> str:
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, str):
        value = value.encode("utf-8")
    return sha256(value).hexdigest()


def documented_missing_field(world: Any, app: str, api: str,
                             arguments: dict, error: str) -> str | None:
    """One required PUBLIC parameter, explicitly named as missing in raw error.

    Never use response body values, candidate solutions or model/ground truth.
    A mere parameter-name mention is insufficient to infer missingness.
    """
    doc = world.task.api_docs.get(app, {}).get(api, {})
    if not isinstance(doc, dict):
        return None
    params = doc.get("parameters", [])
    if not isinstance(params, list):
        return None
    text = str(error)[:4096]
    matched = []
    for item in params:
        if not isinstance(item, dict) or item.get("required") is not True:
            continue
        field = item.get("name")
        if (not isinstance(field, str) or not field.isidentifier()
                or len(field) > 64 or field in arguments):
            continue
        for m in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(field)
                             + r"(?![A-Za-z0-9_])", text):
            window = text[max(0, m.start()-100):m.end()+100]
            if re.search(r"(?i)\b(missing|required|not provided|field required)\b",
                         window):
                matched.append(field)
                break
    return matched[0] if len(set(matched)) == 1 else None


class RecordingModel:
    def __init__(self, actual: Any):
        self.actual = actual
        self.turn_number = 0
        self.recorded: list[dict] = []

    def _complete(self, messages: list[dict], tools: list[dict], final: bool):
        self.turn_number += 1
        answer = (self.actual.complete_final(messages, tools)
                  if final else self.actual.complete(messages, tools))
        self.recorded.append({
            "input_sha": sha({"messages": messages, "tools": tools}),
            "final": final, "reply": deepcopy(answer),
        })
        return answer

    def complete(self, messages, tools):
        return self._complete(messages, tools, False)

    def complete_final(self, messages, tools):
        return self._complete(messages, tools, True)


class PrefixReplayModel:
    def __init__(self, original: RecordingModel, until_turn: int, actual: Any):
        if until_turn < 1 or until_turn > len(original.recorded):
            raise ValueError("invalid original control pivot")
        self.original = original
        self.until_turn = until_turn
        self.actual = actual
        self.turn_number = 0
        self.replayed = 0
        self.live = 0

    def _complete(self, messages, tools, final: bool):
        self.turn_number += 1
        if self.turn_number <= self.until_turn:
            row = self.original.recorded[self.turn_number-1]
            if (row["final"] is not final or
                    row["input_sha"] != sha({"messages": messages, "tools": tools})):
                raise ValueError("protected source / tool feedback prefix drift")
            self.replayed += 1
            return deepcopy(row["reply"])
        self.live += 1
        return (self.actual.complete_final(messages, tools)
                if final else self.actual.complete(messages, tools))

    def complete(self, messages, tools):
        return self._complete(messages, tools, False)

    def complete_final(self, messages, tools):
        return self._complete(messages, tools, True)


class PublicFieldCallTool(AppWorldCallTool):
    """Byte-equivalent original feedback unless ONE documented error field is patched."""

    def __init__(self, world: Any, turn_number, target: dict | None = None):
        super().__init__(world, max_calls=MAX_CALLS, report_http_failures=True)
        self.turn_number = turn_number
        self.target = target
        self.eligible: list[dict] = []
        self.observed: list[dict] = []
        self.patched_count = 0
        self.http_status_counts: dict[str, int] = {}
        self.recovered_original = False
        self._known_failures: list[dict] = []

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        app, api, data = (arguments.get("app"), arguments.get("api"),
                          arguments.get("arguments"))
        if (not isinstance(data, dict) or
                not allowed_public_api(self.world, app, api)):
            return ToolResult(False, "API or arguments not available for this task.",
                              "API_DENIED")
        if self.calls >= self.max_calls:
            return ToolResult(False, "AppWorld evaluation API call budget exhausted.",
                              "CALL_LIMIT")
        self.calls += 1
        ordinal = self.calls
        turn = self.turn_number()
        signature = sha([app, api, data])
        status = None
        missing_field = None
        try:
            response = self.world.requester.request(
                _app_name=app, _api_name=api, show=False,
                raise_on_failure=True, **data)
            result = ToolResult(True, json.dumps(
                response, ensure_ascii=False, sort_keys=True, default=str)[:10000])
            # STRICT: the previously missing field must actually be supplied
            # to the SAME public API and that request must succeed.
            for failure in self._known_failures:
                if (failure["app"] == app and failure["api"] == api
                        and failure["field"] in data
                        and ordinal > failure["ordinal"]):
                    self.recovered_original = True
        except Exception as exc:
            self.errors += 1
            m = re.search(
                r"(?:status code is|response status code is|status code)\s*(\d{3})",
                str(exc), re.I)
            if m is None:
                result = ToolResult(
                    False, "AppWorld request error: " + type(exc).__name__,
                    "APPWORLD_API_ERROR")
            else:
                status = m.group(1)
                self.http_errors += 1
                self.http_status_counts[status] = (
                    self.http_status_counts.get(status, 0) + 1)
                original = FEEDBACK_TEMPLATE.format(status=status)
                missing_field = documented_missing_field(
                    self.world, app, api, data, str(exc))
                if missing_field:
                    eligible = {
                        "ordinal": ordinal, "turn": turn, "signature": signature,
                        "app": app, "api": api, "field": missing_field,
                        "status": status,
                    }
                    self.eligible.append(eligible)
                    self._known_failures.append(eligible)
                result = ToolResult(False, original, "HTTP_ERROR")
                if self.target is not None and ordinal == self.target["ordinal"]:
                    if (missing_field != self.target["field"]
                            or signature != self.target["signature"]
                            or status != self.target["status"]
                            or turn != self.target["turn"]):
                        raise ValueError("single-field fork pivot drift")
                    result = ToolResult(
                        False, original + " Missing required documented API "
                        "parameter: " + missing_field + ".", "HTTP_ERROR")
                    self.patched_count += 1
        full_observation, _ = result.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
        self.observed.append({
            "ordinal": ordinal, "turn": turn, "signature": signature,
            "response_sha": sha(full_observation),
            "ok": result.ok, "status": status,
        })
        return result


def original_prompt(world: Any) -> str:
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


def model() -> OpenAICompatModel:
    return OpenAICompatModel(
        model="qwen3.5-9b", base_url="http://127.0.0.1:8080",
        timeout_s=240, max_new_tokens=384)


def run_task(task_id: str, root: Path) -> dict:
    from appworld import AppWorld
    start = clock()
    with AppWorld(task_id=task_id, experiment_name="private-train-original-feedback-control",
                  ground_truth_mode="full", timeout_seconds=180) as world:
        recorder = RecordingModel(model())
        docs = AppWorldDocsTool(world)
        tool = PublicFieldCallTool(world, lambda: recorder.turn_number)
        original = AgentRuntime(
            recorder, [docs, tool], max_steps=MAX_STEPS).run(
                original_prompt(world), root / "control")
        verdict = world.evaluate(suppress_errors=True)
        baseline_pass = bool(verdict.success)
        baseline_tests = (int(verdict.pass_count), int(verdict.num_tests))
        pivot = tool.eligible[0] if tool.eligible and not baseline_pass else None
        controls = deepcopy(tool.observed)
        baseline_recovered = tool.recovered_original
        baseline_calls = tool.calls
        baseline_errors = tool.http_errors
        baseline_stop = original.stop_reason
        baseline_decisions = recorder.turn_number

    result = {
        "task_hash": sha(task_id), "original_task_success": baseline_pass,
        "original_state_tests_pass": baseline_tests[0],
        "official_state_tests_total": baseline_tests[1],
        "original_stop": baseline_stop, "original_model_turns": baseline_decisions,
        "original_api_calls": baseline_calls, "original_http_errors": baseline_errors,
        "first_http_error_observed": bool(baseline_errors),
        "eligible_public_missing_field_events": len(tool.eligible),
        "eligible_first_blocker": pivot is not None,
        "original_error_resolved_by_original": baseline_recovered,
        "original_generic_feedback_as_seen": None,
        "treatment_task_success": None, "treatment_original_error_resolved": None,
        "changed_only_one_public_feedback_field": False,
        "paired_prefix_verified": False, "paired_task_rescue": False,
        "paired_task_regression": False, "new_adapter": False,
    }
    if not pivot:
        result["fork_status"] = ("original_already_successful"
                                 if baseline_pass else "no_eligible_documented_field")
        result["raw_wall_seconds"] = round((clock()-start)/1e9, 2)
        return result
    result["original_generic_feedback_as_seen"] = (
        "ToolError[HTTP_ERROR]: " + FEEDBACK_TEMPLATE.format(
            status=pivot["status"]))
    result["original_feedback_sha"] = sha(result["original_generic_feedback_as_seen"])
    result["public_field_sha"] = sha(pivot["field"])
    result["fork_turn"] = pivot["turn"]
    result["fork_api_ordinal"] = pivot["ordinal"]
    result["http_status"] = pivot["status"]
    with AppWorld(task_id=task_id, experiment_name="private-train-single-public-field-fork",
                  ground_truth_mode="full", timeout_seconds=180) as world:
        playback = PrefixReplayModel(recorder, pivot["turn"], model())
        docs = AppWorldDocsTool(world)
        patch_tool = PublicFieldCallTool(
            world, lambda: playback.turn_number, target=pivot)
        candidate = AgentRuntime(
            playback, [docs, patch_tool], max_steps=MAX_STEPS).run(
                original_prompt(world), root / "patched")
        new_verdict = world.evaluate(suppress_errors=True)
        # Need exact same API calls/feedback before pivot; no source drift.
        earlier_base = [x for x in controls if x["ordinal"] < pivot["ordinal"]]
        earlier_fork = [x for x in patch_tool.observed
                        if x["ordinal"] < pivot["ordinal"]]
        if (earlier_base != earlier_fork or
                patch_tool.patched_count != 1 or
                playback.replayed != pivot["turn"] or
                int(new_verdict.num_tests) != baseline_tests[1]):
            raise ValueError("protected task state / prefix / official grader drift")
        result.update({
            "fork_status": "complete_paired_fork",
            "paired_prefix_verified": True,
            "changed_only_one_public_feedback_field": True,
            "treatment_task_success": bool(new_verdict.success),
            "treatment_state_tests_pass": int(new_verdict.pass_count),
            "treatment_stop": candidate.stop_reason,
            "treatment_model_turns_live": playback.live,
            "treatment_api_calls": patch_tool.calls,
            "treatment_http_errors": patch_tool.http_errors,
            "treatment_original_error_resolved": patch_tool.recovered_original,
            "paired_task_rescue": not baseline_pass and bool(new_verdict.success),
            "paired_task_regression": baseline_pass and not bool(new_verdict.success),
            "patched_feedback_sha": sha(
                "ToolError[HTTP_ERROR]: " + FEEDBACK_TEMPLATE.format(
                    status=pivot["status"]) +
                " Missing required documented API parameter: " + pivot["field"] + "."),
        })
    result["raw_wall_seconds"] = round((clock()-start)/1e9, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("private-appworld-train-fork"))
    args = parser.parse_args()
    if not 0 <= args.index < 8:
        parser.error("only the already source-locked eight TRAIN scenes are eligible")
    if args.out.exists():
        raise FileExistsError("immutable scored evidence may not be overwritten")
    from appworld import load_task_ids
    from appworld.ground_truth import GroundTruth
    rows = [(tid, GroundTruth.load(task_id=tid, mode="full"))
            for tid in load_task_ids("train")]
    selected = choose_train_entries(rows)
    if len(selected) != 8:
        raise ValueError("source TRAIN8 selection changed")
    try:
        result = run_task(selected[args.index], args.workspace)
    except Exception as exc:
        # Do not dump encrypted-origin task strings, raw API errors or answers.
        raise RuntimeError("PRIVATE_TRAIN_FORK_FAILED:" + type(exc).__name__) from None
    result.update({
        "protocol": PROTOCOL, "split": "train",
        "upstream_commit": UPSTREAM_COMMIT, "model_sha256": GGUF_SHA,
        "source_ids_sha256": sha(sorted(selected)), "index": args.index,
        "task_count": 1, "no_teacher_or_gold_in_model_context": True,
        "protected_task_and_api_values_never_exported": True,
        "no_DEV_TEST_GAIA": True, "no_training": True,
        "claim_boundary": (
            "First public-required-parameter HTTP error in a failed original "
            "TRAIN trajectory is only a candidate blocking point. The changed "
            "field is documented and absent in the original call. Both official "
            "task success and strict same-API recovery are measured separately. "
            "TRAIN task/simulator outcomes cannot be called GAIA uplift.")
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("APPWORLD_TRAIN_FEEDBACK_FORK_SAFE_COUNTS", json.dumps({
        "index": args.index, "original_task_success": result["original_task_success"],
        "eligible": result["eligible_first_blocker"],
        "fork_status": result["fork_status"],
        "treatment_task_success": result["treatment_task_success"],
        "original_error_resolved": result["original_error_resolved_by_original"],
        "treatment_error_resolved": result["treatment_original_error_resolved"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
