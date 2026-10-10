from __future__ import annotations

import json
import random
from pathlib import Path

from .oracle_policy import _base_row, _call, _row_for, _tool_result

_CAPABILITIES = (
    "tool_selection",
    "argument_grounding",
    "multi_step",
    "evidence_to_final",
    "no_tool_stop",
    "failure_recovery",
    "strategy_switch",
    "duplicate_avoidance",
)
_SPLIT_COUNTS = {"train": 1600, "dev": 200, "test": 200}
_PROVENANCE = {
    "license": "CC0-1.0",
    "generator": "small-agent-policy-oracle",
    "generator_version": "2",
    "oracle_type": "deterministic-program",
    "oracle_version": "2",
}


def _upgrade_v1_row(row: dict) -> dict:
    row["source"] = f"synthetic-small-agent-policy-v2/{row['split']}"
    row["provenance"] = dict(_PROVENANCE)
    return row


def _mark_expected_failure(row: dict, error_code: str) -> dict:
    row["verification"]["zero_tool_errors"] = False
    row["verification"]["expected_tool_failures_only"] = True
    row["verification"]["expected_tool_error_codes"] = [error_code]
    row["metrics"]["tool_errors"] = 1
    row["metrics"]["tool_successes"] = max(0, row["metrics"]["tool_calls"] - 1)
    return row


def _failure_recovery_case(rng: random.Random, task_id: str, split: str) -> dict:
    primary = f"PRIMARY-{rng.randrange(100000, 999999)}"
    fallback = f"FALLBACK-{rng.randrange(100000, 999999)}"
    code = f"CODE-{rng.randrange(100000, 999999)}"
    observation = (
        f"[1] Fallback release bulletin\nURL: https://example.invalid/{fallback}\n"
        f"The release code is {code}."
    )
    question = (
        f"Find the release code. First search for {primary}. If that search has no usable result, "
        f"recover by searching for fallback alias {fallback}. Return only the code once found."
    )
    first_id = f"{task_id}-call-1"
    second_id = f"{task_id}-call-2"
    row = _base_row(
        task_id=task_id,
        split=split,
        capability="failure_recovery",
        question=question,
        expected=code,
        required_tools=["search"],
        messages=[
            _call(first_id, "search", {"query": primary, "max_results": 5}),
            _tool_result(first_id, "search", "ERROR[SEARCH_FIXTURE_MISS]: No deterministic search fixture matched query"),
            _call(second_id, "search", {"query": fallback, "max_results": 5}),
            _tool_result(second_id, "search", observation),
            {"role": "assistant", "content": code},
        ],
        search_fixtures={fallback: observation},
    )
    return _mark_expected_failure(_upgrade_v1_row(row), "SEARCH_FIXTURE_MISS")


def _strategy_switch_case(rng: random.Random, task_id: str, split: str) -> dict:
    project = f"PROJECT-{rng.randrange(100000, 999999)}"
    token = f"TOKEN-{rng.randrange(100000, 999999)}"
    filename = f"fallback-{rng.randrange(100000, 999999)}.txt"
    content = f"Fallback record for {project}: {token}\n"
    question = (
        f"Find the token for {project}. Try web search first. If search fails, switch strategy and read {filename}. "
        "Return only the token."
    )
    search_id = f"{task_id}-call-1"
    read_id = f"{task_id}-call-2"
    row = _base_row(
        task_id=task_id,
        split=split,
        capability="strategy_switch",
        question=question,
        expected=token,
        required_tools=["search", "read"],
        messages=[
            _call(search_id, "search", {"query": f"{project} token", "max_results": 5}),
            _tool_result(search_id, "search", "ERROR[SEARCH_FIXTURE_MISS]: No deterministic search fixture matched query"),
            _call(read_id, "read", {"source": filename}),
            _tool_result(read_id, "read", content),
            {"role": "assistant", "content": token},
        ],
        files={filename: content},
        search_fixtures={"NEVER-MATCH-THIS-SYNTHETIC-QUERY": "unused fixture"},
    )
    return _mark_expected_failure(_upgrade_v1_row(row), "SEARCH_FIXTURE_MISS")


def _duplicate_avoidance_case(rng: random.Random, task_id: str, split: str) -> dict:
    project = f"ONCE-{rng.randrange(100000, 999999)}"
    value = f"VALUE-{rng.randrange(100000, 999999)}"
    observation = f"[1] Result\nURL: https://example.invalid/{project}\nThe requested value is {value}."
    question = (
        f"Search once for the value of {project}. When the evidence gives the value, do not repeat the same call; "
        "stop using tools and return only the value."
    )
    call_id = f"{task_id}-call-1"
    row = _base_row(
        task_id=task_id,
        split=split,
        capability="duplicate_avoidance",
        question=question,
        expected=value,
        required_tools=["search"],
        messages=[
            _call(call_id, "search", {"query": f"{project} value", "max_results": 5}),
            _tool_result(call_id, "search", observation),
            {"role": "assistant", "content": value},
        ],
        search_fixtures={project: observation},
    )
    return _upgrade_v1_row(row)


def _row_for_v2(capability: str, rng: random.Random, task_id: str, split: str, index: int) -> dict:
    if capability == "failure_recovery":
        return _failure_recovery_case(rng, task_id, split)
    if capability == "strategy_switch":
        return _strategy_switch_case(rng, task_id, split)
    if capability == "duplicate_avoidance":
        return _duplicate_avoidance_case(rng, task_id, split)
    return _upgrade_v1_row(_row_for(capability, rng, task_id, split, index))


def generate_oracle_policy_v2_dataset(
    output_dir: str | Path,
    *,
    seed: str = "small-agent-policy-v2",
) -> dict[str, Path]:
    """Generate deterministic GAIA-independent QLoRA v2 policy trajectories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split, count in _SPLIT_COUNTS.items():
        if count % len(_CAPABILITIES) != 0:
            raise RuntimeError(f"split {split} cannot be balanced across capabilities")
        per_capability = count // len(_CAPABILITIES)
        rows: list[dict] = []
        for capability in _CAPABILITIES:
            rng = random.Random(f"{seed}:{split}:{capability}")
            for index in range(per_capability):
                task_id = f"small-agent-policy-v2-{split}-{capability}-{index + 1:04d}"
                rows.append(_row_for_v2(capability, rng, task_id, split, index))

        path = output_dir / f"{split}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different oracle dataset: {path}")
        path.write_text(text, encoding="utf-8")
        paths[split] = path

    return paths


__all__ = ["generate_oracle_policy_v2_dataset"]
