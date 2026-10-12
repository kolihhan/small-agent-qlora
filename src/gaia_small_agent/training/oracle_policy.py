from __future__ import annotations

import json
import random
from pathlib import Path

from ..agent.loop import SYSTEM_PROMPT
from ..tools import default_tools

_CAPABILITIES = (
    "tool_selection",
    "argument_grounding",
    "multi_step",
    "evidence_to_final",
    "no_tool_stop",
)
_SPLIT_COUNTS = {"train": 1600, "dev": 200, "test": 200}
_PROVENANCE = {
    "license": "CC0-1.0",
    "generator": "tinyagent-policy-oracle",
    "generator_version": "1",
    "oracle_type": "deterministic-program",
    "oracle_version": "1",
}


def _tool_schemas() -> list[dict]:
    return [tool.definition() for tool in default_tools()]


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _tool_result(call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def _base_row(
    *,
    task_id: str,
    split: str,
    capability: str,
    question: str,
    expected: str,
    required_tools: list[str],
    messages: list[dict],
    files: dict[str, str] | None = None,
    search_fixtures: dict[str, str] | None = None,
) -> dict:
    tool_calls = sum(len(message.get("tool_calls") or []) for message in messages)
    return {
        "task_id": task_id,
        "source": f"synthetic-tinyagent-policy-v1/{split}",
        "split": split,
        "capability": capability,
        "verified": True,
        "verification": {
            "completed": True,
            "final_answer_correct": True,
            "zero_tool_errors": True,
            "zero_duplicate_blocks": True,
            "required_tools_satisfied": True,
            "provenance_complete": True,
        },
        "provenance": dict(_PROVENANCE),
        "required_tools": list(required_tools),
        "expected_answer": expected,
        "tools": _tool_schemas(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            *messages,
        ],
        "metrics": {
            "steps": max(1, tool_calls),
            "tool_calls": tool_calls,
            "tool_successes": tool_calls,
            "tool_errors": 0,
            "duplicate_calls_blocked": 0,
        },
        "eval_task": {
            "question": question,
            "expected_answer": expected,
            "required_tools": list(required_tools),
            "files": files or {},
            "search_fixtures": search_fixtures or {},
            "capability": capability,
        },
    }


def _single_tool_case(rng: random.Random, task_id: str, split: str, capability: str, index: int) -> dict:
    kind = ("python", "read", "inspect", "search")[index % 4]
    call_id = f"{task_id}-call-1"

    if kind == "python":
        a, b = rng.randrange(17, 999), rng.randrange(11, 97)
        expected = str(a * b)
        question = f"Calculate {a} * {b} exactly. Use the best available tool and return only the integer."
        messages = [
            _call(call_id, "python", {"code": f"print({a} * {b})"}),
            _tool_result(call_id, "python", expected),
            {"role": "assistant", "content": expected},
        ]
        return _base_row(
            task_id=task_id,
            split=split,
            capability=capability,
            question=question,
            expected=expected,
            required_tools=["python"],
            messages=messages,
        )

    if kind == "read":
        key = f"KEY_{rng.randrange(100000, 999999)}"
        value = f"VALUE_{rng.randrange(100000, 999999)}"
        filename = f"note-{rng.randrange(100000, 999999)}.txt"
        content = f"IGNORE=alpha\n{key}={value}\nIGNORE=omega\n"
        question = f"Read {filename} and return only the value for {key}."
        messages = [
            _call(call_id, "read", {"source": filename}),
            _tool_result(call_id, "read", content),
            {"role": "assistant", "content": value},
        ]
        return _base_row(
            task_id=task_id,
            split=split,
            capability=capability,
            question=question,
            expected=value,
            required_tools=["read"],
            messages=messages,
            files={filename: content},
        )

    if kind == "inspect":
        filename = f"table-{rng.randrange(100000, 999999)}.csv"
        columns = 3 + rng.randrange(0, 4)
        header = ",".join(f"c{i}" for i in range(1, columns + 1))
        values = ",".join(str(rng.randrange(1, 50)) for _ in range(columns))
        content = f"{header}\n{values}\n"
        expected = str(columns)
        question = f"Inspect {filename} and return only the number of columns."
        observation = f"CSV metadata: rows=1, columns={columns}, headers=[{header}]"
        messages = [
            _call(call_id, "inspect", {"path": filename}),
            _tool_result(call_id, "inspect", observation),
            {"role": "assistant", "content": expected},
        ]
        return _base_row(
            task_id=task_id,
            split=split,
            capability=capability,
            question=question,
            expected=expected,
            required_tools=["inspect"],
            messages=messages,
            files={filename: content},
        )

    project = f"PROJECT-{rng.randrange(100000, 999999)}"
    code = f"CODE-{rng.randrange(100000, 999999)}"
    query = f"{project} release code"
    question = f"Search the public web for the release code of {project}. Return only the code."
    observation = f"[1] {project} release bulletin\nURL: https://example.invalid/{project}\nThe release code is {code}."
    messages = [
        _call(call_id, "search", {"query": query, "max_results": 5}),
        _tool_result(call_id, "search", observation),
        {"role": "assistant", "content": code},
    ]
    return _base_row(
        task_id=task_id,
        split=split,
        capability=capability,
        question=question,
        expected=code,
        required_tools=["search"],
        messages=messages,
        search_fixtures={project: observation},
    )


def _multi_step_case(rng: random.Random, task_id: str, split: str) -> dict:
    a, b = rng.randrange(20, 500), rng.randrange(20, 500)
    filename = f"numbers-{rng.randrange(100000, 999999)}.txt"
    content = f"left={a}\nright={b}\n"
    expected = str(a + b)
    question = f"Read {filename}, add the values named left and right using Python, and return only the sum."
    read_id = f"{task_id}-call-1"
    python_id = f"{task_id}-call-2"
    messages = [
        _call(read_id, "read", {"source": filename}),
        _tool_result(read_id, "read", content),
        _call(python_id, "python", {"code": f"print({a} + {b})"}),
        _tool_result(python_id, "python", expected),
        {"role": "assistant", "content": expected},
    ]
    return _base_row(
        task_id=task_id,
        split=split,
        capability="multi_step",
        question=question,
        expected=expected,
        required_tools=["read", "python"],
        messages=messages,
        files={filename: content},
    )


def _evidence_to_final_case(rng: random.Random, task_id: str, split: str, index: int) -> dict:
    if index % 2 == 0:
        token = f"TOKEN_{rng.randrange(100000, 999999)}"
        filename = f"evidence-{rng.randrange(100000, 999999)}.txt"
        content = f"The requested token is {token}.\n"
        question = f"Read {filename}. Once the requested token is known, stop using tools and return only the token."
        call_id = f"{task_id}-call-1"
        messages = [
            _call(call_id, "read", {"source": filename}),
            _tool_result(call_id, "read", content),
            {"role": "assistant", "content": token},
        ]
        return _base_row(
            task_id=task_id,
            split=split,
            capability="evidence_to_final",
            question=question,
            expected=token,
            required_tools=["read"],
            messages=messages,
            files={filename: content},
        )

    project = f"FACT-{rng.randrange(100000, 999999)}"
    value = f"VALUE-{rng.randrange(100000, 999999)}"
    observation = f"[1] Fact page\nURL: https://example.invalid/{project}\n{project} has value {value}."
    question = f"Search for the value of {project}. Once found, stop searching and return only the value."
    call_id = f"{task_id}-call-1"
    messages = [
        _call(call_id, "search", {"query": f"{project} value", "max_results": 5}),
        _tool_result(call_id, "search", observation),
        {"role": "assistant", "content": value},
    ]
    return _base_row(
        task_id=task_id,
        split=split,
        capability="evidence_to_final",
        question=question,
        expected=value,
        required_tools=["search"],
        messages=messages,
        search_fixtures={project: observation},
    )


def _no_tool_case(rng: random.Random, task_id: str, split: str) -> dict:
    value = f"DIRECT-{rng.randrange(100000, 999999)}"
    question = f"The answer token is {value}. Return only that token. Do not use a tool."
    return _base_row(
        task_id=task_id,
        split=split,
        capability="no_tool_stop",
        question=question,
        expected=value,
        required_tools=[],
        messages=[{"role": "assistant", "content": value}],
    )


def _row_for(capability: str, rng: random.Random, task_id: str, split: str, index: int) -> dict:
    if capability in {"tool_selection", "argument_grounding"}:
        return _single_tool_case(rng, task_id, split, capability, index)
    if capability == "multi_step":
        return _multi_step_case(rng, task_id, split)
    if capability == "evidence_to_final":
        return _evidence_to_final_case(rng, task_id, split, index)
    if capability == "no_tool_stop":
        return _no_tool_case(rng, task_id, split)
    raise ValueError(f"unknown capability: {capability}")


def generate_oracle_policy_dataset(
    output_dir: str | Path,
    *,
    seed: str = "tinyagent-policy-v1",
) -> dict[str, Path]:
    """Generate deterministic, GAIA-independent TinyAgent-style policy trajectories."""
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
                task_id = f"tinyagent-policy-v1-{split}-{capability}-{index + 1:04d}"
                rows.append(_row_for(capability, rng, task_id, split, index))

        path = output_dir / f"{split}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different oracle dataset: {path}")
        path.write_text(text, encoding="utf-8")
        paths[split] = path

    return paths


__all__ = ["generate_oracle_policy_dataset"]
