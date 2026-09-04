from __future__ import annotations

import json
import random
from pathlib import Path

_CATEGORIES = ("read_fact", "inspect_metadata", "python_numeric", "format_recovery")


def _base_row(*, task_id: str, category: str, question: str, expected: str, required_tools: list[str], files: dict[str, str] | None = None) -> dict:
    return {
        "id": task_id,
        "category": category,
        "source": "synthetic-policy-v1",
        "license": "CC0-1.0",
        "generator": "p4-policy-tasks",
        "generator_version": "1",
        "oracle_type": "exact",
        "oracle_version": "1",
        "question": question,
        "expected_answer": expected,
        "required_tools": required_tools,
        "files": files or {},
    }


def generate_policy_tasks(output_path: str | Path, *, count: int = 64, seed: str = "p4-policy-v1") -> Path:
    """Generate a small deterministic, non-GAIA policy curriculum.

    The tasks exercise only generic local capabilities and carry their own exact
    oracle and provenance. They are intentionally independent of GAIA question
    text and are frozen before the sealed evaluation partition is inspected.
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
                task_id=task_id, category=category,
                question=f"Read {filename}. Return only the value for {key}.",
                expected=value, required_tools=["read"],
                files={filename: f"ALPHA=ignore\n{key}={value}\nOMEGA=ignore\n"},
            ))
        elif category == "inspect_metadata":
            filename = f"table-{index + 1:03d}.csv"
            rows.append(_base_row(
                task_id=task_id, category=category,
                question=f"Inspect {filename}. Return only the number of columns.",
                expected="3", required_tools=["inspect"],
                files={filename: "name,score,status\na,10,ok\nb,20,ok\n"},
            ))
        elif category == "python_numeric":
            a, b = rng.randrange(11, 80), rng.randrange(11, 80)
            expected = str(a * b)
            rows.append(_base_row(
                task_id=task_id, category=category,
                question=f"Use the python tool to compute {a} * {b}. Return only the integer result.",
                expected=expected, required_tools=["python"],
            ))
        else:
            a, b, c = rng.randrange(2, 20), rng.randrange(2, 20), rng.randrange(2, 10)
            expected = str((a + b) * c)
            rows.append(_base_row(
                task_id=task_id, category=category,
                question=(f"Use the python tool to compute ({a} + {b}) * {c}. "
                          "If an action fails, change strategy rather than repeating it. Return only the integer result."),
                expected=expected, required_tools=["python"],
            ))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite different policy task set: {output_path}")
    output_path.write_text(text, encoding="utf-8")
    return output_path
