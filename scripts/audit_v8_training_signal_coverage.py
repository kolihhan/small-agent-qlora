"""Label-blind CPU audit of *frozen* Qwen3.5-9B v8 next-action SFT exposure.

Purpose: distinguish supervision of a multi-hop source trace from supervision
of *recovering* from agent mistakes. Reads original TRAIN+HOLDOUT artifacts;
emits ONLY aggregate counts, source identifiers, pinned hashes and limitations.
Never export user/task text, answer content, tool observations or GAIA data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

TRAIN_SHA256 = "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
EXPECTED_PARENTS = (304, 35)
EXPECTED_ACTIONS = (798, 176)
EXPECTED_SOURCES = {
    "synthetic-small-agent-policy-v3/train",
    "2wikimultihopqa-train",
    "hotpotqa-distractor-train",
    "finqa-train",
    "musique-train",
}
ERROR_TOKENS = ("toolerror[", "[error]", "tool error:", "error_code:")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_tool_error(message: dict) -> bool:
    content = str(message.get("content") or "").casefold()
    return (any(marker in content for marker in ERROR_TOKENS)
            or message.get("ok") is False
            or message.get("error_code") is not None)


def _has_call(message: dict) -> bool:
    return bool(message.get("tool_calls"))


def _selected(row: dict) -> list[int]:
    messages = row.get("messages")
    indices = row.get("training_action_indices")
    if (not isinstance(messages, list) or not messages
        or not isinstance(indices, list) or not indices
        or len(indices) != len(set(indices)) or indices != sorted(indices)
        or any(type(i) is not int or i < 0 or i >= len(messages)
               or messages[i].get("role") != "assistant" for i in indices)):
        raise ValueError("Invalid original selected assistant next-action indices")
    return indices


def _source_group(source: str) -> str:
    if source == "musique-train":
        return "musique_annotation_oracle"
    if source == "finqa-train":
        return "finqa_observed"
    if source in {"2wikimultihopqa-train", "hotpotqa-distractor-train"}:
        return "multihop_observed"
    if source == "synthetic-small-agent-policy-v3/train":
        return "synthetic_policy"
    raise ValueError("Unexpected split or SFT source (possible leakage)")


def audit_rows(path: Path, *, expected_parents: int, expected_actions: int,
               expected_digest: str | None = None) -> tuple[dict, set[str], set[str]]:
    if expected_digest is not None and sha(path) != expected_digest:
        raise ValueError("Frozen training/holdout content SHA changed")
    grouped: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    hashes = set()
    ids = set()
    source_counts = Counter()
    with path.open(encoding="utf8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ident = str(row.get("task_id") or "")
            if not ident or ident in ids:
                raise ValueError("Missing or duplicate task parent")
            ids.add(ident)
            source = str(row.get("source") or "")
            if source not in EXPECTED_SOURCES:
                raise ValueError("Unexpected source/split")
            if row.get("verified") is not True:
                raise ValueError("Unverified source row")
            if _source_group(source) == "musique_annotation_oracle":
                if row.get("derived_from_annotation_not_observed_model") is not True:
                    raise ValueError("Annotated source provenance missing")
            question = [m.get("content") for m in row.get("messages", [])
                        if isinstance(m, dict) and m.get("role") == "user"]
            if len(question) != 1 or not isinstance(question[0], str):
                raise ValueError("No unique task instruction")
            qhash = hashlib.sha256(" ".join(question[0].split()).casefold().encode()).hexdigest()
            if qhash in hashes:
                raise ValueError("Duplicate question fingerprint")
            hashes.add(qhash)
            indices = _selected(row)
            messages = row["messages"]
            c = grouped[_source_group(source)]
            c["parents"] += 1
            source_counts[source] += 1
            tool_steps = sum(m.get("role") == "tool" for m in messages)
            if tool_steps >= 2:
                c["parents_with_2plus_tool_observations"] += 1
            if tool_steps >= 4:
                c["parents_with_4plus_tool_observations"] += 1
            if any(_has_tool_error(m) for m in messages if m.get("role") == "tool"):
                c["parents_with_any_tool_error"] += 1
            if row.get("derived_from_annotation_not_observed_model") is True:
                c["annotation_not_replayed_model_trajectories"] += 1

            for idx in indices:
                msg = messages[idx]
                c["selected_actions"] += 1
                has_call = _has_call(msg)
                c["selected_tool_action" if has_call else "selected_final_action"] += 1
                prefix = messages[:idx]
                tool_history = [m for m in prefix if m.get("role") == "tool"]
                prior = prefix[-1] if prefix else {}
                prev_error = prior.get("role") == "tool" and _has_tool_error(prior)
                if has_call and tool_history:
                    c["selected_tool_action_after_observation"] += 1
                if has_call and prev_error:
                    c["selected_tool_action_directly_after_tool_error"] += 1
                if not has_call and tool_history:
                    c["selected_final_after_tool_observation"] += 1
                if not has_call and len(tool_history) >= 2:
                    c["selected_final_after_2plus_observations"] += 1
                if not has_call and len(tool_history) >= 4:
                    c["selected_final_after_4plus_observations"] += 1
                if not has_call and prev_error:
                    c["selected_final_directly_after_tool_error"] += 1
                if has_call and not tool_history:
                    c["selected_tool_first_step"] += 1
                if has_call and tool_history and not prev_error:
                    c["selected_tool_after_nonerror_observation"] += 1
                if idx == indices[-1] and not has_call:
                    c["selected_last_action_final"] += 1
    if len(ids) != expected_parents:
        raise ValueError(f"Unexpected SFT parent count {len(ids)}")
    total = sum(c["selected_actions"] for c in grouped.values())
    if total != expected_actions:
        raise ValueError(f"Unexpected selected action count {total}")
    for c in grouped.values():
        totals.update(c)
    return {
        "n_parents": len(ids), "n_supervised_actions": total,
        "sha256": sha(path), "per_source": {
            group: dict(sorted(c.items())) for group, c in sorted(grouped.items())
        },
        "source_parent_counts": dict(sorted(source_counts.items())),
        "overall": dict(sorted(totals.items())),
    }, ids, hashes


def audit(train: Path, holdout: Path, manifest: Path) -> dict:
    m = json.loads(manifest.read_text(encoding="utf8"))
    if (m.get("train_parents") != EXPECTED_PARENTS[0]
        or m.get("validation_parents") != EXPECTED_PARENTS[1]
        or m.get("train_actions") != EXPECTED_ACTIONS[0]
        or m.get("validation_actions") != EXPECTED_ACTIONS[1]
        or m.get("train_sha256") != TRAIN_SHA256
        or not isinstance(m.get("validation_sha256"), str)
        or m.get("no_gaia_training") is not True):
        raise ValueError("Frozen original v8 manifest audit mismatch")
    tr, ids, hashes = audit_rows(
        train, expected_parents=304, expected_actions=798, expected_digest=TRAIN_SHA256)
    held, held_ids, held_hashes = audit_rows(
        holdout, expected_parents=35, expected_actions=176,
        expected_digest=m["validation_sha256"])
    if ids & held_ids or hashes & held_hashes:
        raise ValueError("Train/holdout parent or question leakage")
    return {
        "protocol": "qlora-v8-train-signal-coverage-audit-v1",
        "model": "Qwen/Qwen3.5-9B",
        "v8_training_run": 44925307580,
        "source_preflight_run": 43619541894,
        "train": tr, "holdout": held,
        "parent_overlap": 0, "question_overlap": 0,
        "task_question_or_answer_exported": False,
        "gaia_question_or_gold_accessed": False,
        "no_gpu_or_model_inference": True,
        "limits": [
            "Source annotations are teacher-generated next-actions, not successful autonomous agent rollouts.",
            "An error-token heuristic can undercount silent tool failure; report is aggregate only.",
            "Training action exposure is not a causal explanation of the model's GAIA failures.",
            "Multiple prior tool observations do not guarantee a valid multi-hop reasoning chain.",
        ]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = audit(a.train, a.holdout, a.manifest)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError("No overwrite of frozen audit")
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("QLORA_V8_TRAIN_SIGNAL_COVERAGE",json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
