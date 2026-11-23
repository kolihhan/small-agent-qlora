"""One-variable v9 QLoRA *data-only* hypothesis: real-source error-conditioned next actions.

Replace the input context (NOT the target action, task, final answer, assistant
supervision count, tool schema or training-parent order) for one source-grounded
read action in some genuine TRAIN trajectories. Simulate a missing read via the
actual ReadTool against the task's files; next selected action is the source's
original correct read. This is a SYNTHETIC FAILURE on a REAL-SOURCE TASK, not
an observed model recovery. No GAIA, MuSiQue DEV or heldout source is used.

This script prepares a candidate corpus and provenance report. It does NOT
train a model or claim improved accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files

ORIGINAL_TRAIN_SHA = "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
SEED = "step100-qlora-v9-realsource-tool-miss-20261123-fixed-v1"
ELIGIBLE = {"2wikimultihopqa-train", "hotpotqa-distractor-train", "finqa-train"}
EXPECTED_PARENTS, EXPECTED_ACTIONS = 304, 798


def _serialized_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]


def _read_action(message: dict) -> str | None:
    if message.get("role") != "assistant":
        return None
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or not isinstance(calls[0], dict):
        return None
    func = calls[0].get("function") or {}
    if func.get("name") != "read":
        return None
    args = func.get("arguments") or {}
    if not isinstance(args, dict):
        return None
    src = args.get("source")
    if not isinstance(src, str) or not src or len(src) > 250:
        return None
    return src


def _visible_before(messages: list[dict], idx: int, path: str) -> bool:
    # The target path must already be visible, NOT oracle-selected from gold
    # or exposed only in a later observation. This protects causal inference.
    prefix = messages[:idx]
    visible = "\n".join(str(x.get("content") or "") for x in prefix
                        if x.get("role") in {"user", "tool"})
    return path in visible


def augment(row: dict) -> tuple[dict, str] | None:
    if row.get("source") not in ELIGIBLE:
        return None
    task = row.get("eval_task") or {}
    files = task.get("files") or {}
    if not isinstance(files, dict) or not files:
        return None
    messages = row.get("messages") or []
    selected = row.get("training_action_indices") or []
    if not isinstance(selected, list) or not selected:
        raise ValueError("Unverified original TRAIN action indices")
    read_tool = next(t for t in default_tools() if t.name == "read")
    for idx in selected:
        target = _read_action(messages[idx])
        if target is None or target not in files or not _visible_before(messages, idx, target):
            continue
        ident = str(row["task_id"])
        suffix = sha256(f"{SEED}:{ident}:{idx}".encode()).hexdigest()[:12]
        bad = target + ".v9-missing-" + suffix
        if bad in files:
            raise ValueError("Injected-missing path already exists")
        invalid_id = "v9_failed_read_" + suffix
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _materialize_files({"files": files}, root)
            arguments = {"source": bad}
            if read_tool.validate_arguments(arguments) is not None:
                raise ValueError("Injected read invalid under native schema")
            result = read_tool.run(arguments, root)
            if result.ok or not result.error_code:
                raise ValueError("ReadTool did not verify expected missing-file failure")
            observed, _ = result.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
        if "ToolError[" not in observed or bad not in observed:
            raise ValueError("Error observation does not identify failed source")
        injected = [
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": invalid_id,
                    "type": "function",
                    "function": {"name": "read", "arguments": arguments},
                }],
            },
            {
                "role": "tool", "tool_call_id": invalid_id,
                "name": "read", "content": observed,
            }
        ]
        new_messages = messages[:idx] + injected + messages[idx:]
        new_indices = [j if j < idx else j + len(injected) for j in selected]
        if len(new_indices) != len(selected):
            raise AssertionError("Action count drift")
        cloned = dict(row)
        cloned["messages"] = new_messages
        cloned["training_action_indices"] = new_indices
        verification = dict(row.get("verification") or {})
        if verification.get("zero_tool_errors") is not True:
            raise ValueError("Expected clean original tool trace for source augmentation")
        verification["zero_tool_errors"] = False
        verification["expected_tool_failures_only"] = True
        verification["expected_tool_error_codes"] = [result.error_code]
        cloned["verification"] = verification
        metrics = dict(row.get("metrics") or {})
        metrics["tool_calls"] = int(metrics.get("tool_calls") or 0) + 1
        metrics["tool_errors"] = int(metrics.get("tool_errors") or 0) + 1
        cloned["metrics"] = metrics
        cloned["counterfactual_tool_failure"] = {
            "schema": "v9-causally-observed-tool-missing-v1",
            "failure_code": result.error_code,
            "not_observed_agent_recovery": True,
            "read_error_physically_executed": True,
            "original_target_unchanged": True,
            "visible_filename_before_action": True,
            "new_supervised_actions": 0,
        }
        return cloned, row["source"]
    return None


def _ineligibility_reason(row: dict) -> str:
    task=row.get("eval_task") or {}
    files=task.get("files") or {}
    if not isinstance(files,dict) or not files:
        return "no_materialized_files"
    msgs=row.get("messages") or []
    selected=row.get("training_action_indices") or []
    targets=[(i,_read_action(msgs[i])) for i in selected]
    targets=[(i,t) for i,t in targets if t is not None]
    if not targets:
        return "no_selected_read_actions"
    exists=[(i,t) for i,t in targets if t in files]
    if not exists:
        return "selected_read_sources_not_in_eval_files"
    if not any(_visible_before(msgs,i,t) for i,t in exists):
        return "read_paths_not_visible_prior_to_action"
    return "source_replay_failed_before_selection"


def build(train: Path, heldout: Path, manifest: Path) -> tuple[list[dict], dict]:
    source_sha = sha256(train.read_bytes()).hexdigest()
    if source_sha != ORIGINAL_TRAIN_SHA:
        raise ValueError("Frozen original v8 TRAIN SHA mismatch")
    report = json.loads(manifest.read_text(encoding="utf8"))
    if (report.get("train_sha256") != ORIGINAL_TRAIN_SHA
        or report.get("no_gaia_training") is not True
        or report.get("train_actions") != EXPECTED_ACTIONS
        or report.get("train_parents") != EXPECTED_PARENTS
        or sha256(heldout.read_bytes()).hexdigest() != report.get("validation_sha256")):
        raise ValueError("Original v8 source manifest/holdout mismatch")
    rows = _serialized_rows(train)
    hold = _serialized_rows(heldout)
    if len(rows) != 304 or len(hold) != 35:
        raise ValueError("Unexpected source parent count")
    original_ids = [r["task_id"] for r in rows]
    if len(set(original_ids)) != len(rows) or set(original_ids) & {r["task_id"] for r in hold}:
        raise ValueError("Original training/validation identity leakage")
    out = []
    augmented_source = Counter()
    rejected = Counter()
    for row in rows:
        if "gaia" in str(row.get("source","")).casefold():
            raise ValueError("GAIA training data is forbidden")
        result = augment(row)
        if result is None:
            out.append(row)
            if row.get("source") in ELIGIBLE:
                rejected[str(row["source"])+":"+_ineligibility_reason(row)] += 1
        else:
            new, source = result
            out.append(new)
            augmented_source[source] += 1
    if [x["task_id"] for x in out] != original_ids:
        raise ValueError("Changed source training order or parent selection")
    if sum(len(r["training_action_indices"]) for r in out) != EXPECTED_ACTIONS:
        raise ValueError("Altered overall number of SFT actions")
    for old, new in zip(rows, out):
        if (old.get("expected_answer") != new.get("expected_answer")
            or old.get("eval_task") != new.get("eval_task")
            or old.get("tools") != new.get("tools")):
            raise ValueError("Modified task, final answer or tool schema")
        old_targets = [old["messages"][i] for i in old["training_action_indices"]]
        new_targets = [new["messages"][i] for i in new["training_action_indices"]]
        if old_targets != new_targets:
            raise ValueError("Changed supervised next-action labels")
    n = sum(augmented_source.values())
    if n < 12:
        raise ValueError(f"Insufficient grounded real-source recovery interventions: {n}; rejection_categories={dict(sorted(rejected.items()))}")
    return out, {
        "protocol": "step100-v9-real-source-error-conditioned-matched-action-data-v1",
        "original_train_sha256": source_sha,
        "n_parents": len(out), "n_original_supervised_actions": EXPECTED_ACTIONS,
        "n_candidate_supervised_actions": EXPECTED_ACTIONS,
        "n_injected_missing_file_observations": n,
        "injected_real_source_parent_counts": dict(sorted(augmented_source.items())),
        "eligible_without_grounded_path": dict(sorted(rejected.items())),
        "source_parent_order_preserved": True,
        "supervised_action_labels_identical": True,
        "original_task_answers_unchanged": True,
        "original_holdout_unmodified_sha256": report["validation_sha256"],
        "any_gaia_input_or_gold_used": False,
        "gpu_training_ran": False,
        "limitations": [
            "This is a synthetic file-miss injection on genuine training-source tasks, NOT observed model error recovery.",
            "Creating counterfactual source episodes cannot by itself establish a generalization improvement.",
            "All source training questions and answer labels are held out of terminal logs; a separate matched training and unseen-source evaluation are required.",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.report.exists():
        raise FileExistsError("Refuse to overwrite candidate training artifacts")
    data, proof=build(a.train,a.holdout,a.manifest)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in data))
    proof["candidate_train_sha256"]=sha256(a.output.read_bytes()).hexdigest()
    a.report.write_text(json.dumps(proof,sort_keys=True,indent=2)+"\n")
    print("V9_REAL_SOURCE_RECOVERY_DATA_PREFLIGHT",json.dumps(proof,sort_keys=True))


if __name__ == "__main__":
    main()
