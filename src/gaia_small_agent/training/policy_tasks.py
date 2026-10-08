from __future__ import annotations

import json
import random
from pathlib import Path

_CATEGORIES = ("read_fact", "inspect_metadata", "python_numeric", "read_then_python")


def _base_row(*, task_id: str, category: str, question: str, expected: str, required_tools: list[str], files: dict[str, str] | None = None) -> dict:
    return {
        "id": task_id,
        "category": category,
        "source": "synthetic-policy-v2",
        "license": "CC0-1.0",
        "generator": "p4-policy-tasks",
        "generator_version": "2",
        "oracle_type": "exact",
        "oracle_version": "1",
        "question": question,
        "expected_answer": expected,
        "required_tools": required_tools,
        "files": files or {},
    }


def generate_policy_tasks(output_path: str | Path, *, count: int = 64, seed: str = "p4-policy-v2") -> Path:
    """Generate a deterministic, non-GAIA policy curriculum.

    Prompts describe the goal instead of naming the tool. One quarter of the
    curriculum requires a real two-step read -> python trajectory. This keeps
    the training set small while exercising autonomous tool selection rather
    than simple compliance with an explicit tool instruction.
    """
    if count < 4 or count > 128 or count % len(_CATEGORIES) != 0:
        raise ValueError("count must be a multiple of 4 between 4 and 128")
    rng = random.Random(seed)
    rows: list[dict] = []
    for index in range(count):
        category = _CATEGORIES[index % len(_CATEGORIES)]
        task_id = f"policy-{index + 1:03d}"
        if category == "read_fact":
            key = f"CODE_{index + 1:03d}"
            value = f"VALUE_{rng.randrange(1000, 9999)}"
            filename = f"facts-{index + 1:03d}.txt"
            rows.append(_base_row(
                task_id=task_id,
                category=category,
                question=f"What value is assigned to {key} in {filename}? Return only the value.",
                expected=value,
                required_tools=["read"],
                files={filename: f"ALPHA=ignore\n{key}={value}\nOMEGA=ignore\n"},
            ))
        elif category == "inspect_metadata":
            filename = f"table-{index + 1:03d}.csv"
            rows.append(_base_row(
                task_id=task_id,
                category=category,
                question=f"How many columns does {filename} contain? Return only the number.",
                expected="3",
                required_tools=["inspect"],
                files={filename: "name,score,status\na,10,ok\nb,20,ok\n"},
            ))
        elif category == "python_numeric":
            a, b = rng.randrange(10_000, 90_000), rng.randrange(10_000, 90_000)
            expected = str(a * b)
            rows.append(_base_row(
                task_id=task_id,
                category=category,
                question=f"Return the exact integer product of {a} and {b}.",
                expected=expected,
                required_tools=["python"],
            ))
        else:
            a, b = rng.randrange(1_000, 9_000), rng.randrange(1_000, 9_000)
            expected = str(a * b)
            filename = f"operands-{index + 1:03d}.txt"
            rows.append(_base_row(
                task_id=task_id,
                category=category,
                question=(
                    f"Using the A and B values stored in {filename}, return their exact product as an integer."
                ),
                expected=expected,
                required_tools=["read", "python"],
                files={filename: f"A={a}\nB={b}\n"},
            ))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite different policy task set: {output_path}")
    output_path.write_text(text, encoding="utf-8")
    return output_path
