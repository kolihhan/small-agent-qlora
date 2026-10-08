from __future__ import annotations

import json
import random
from pathlib import Path

_CATEGORIES = ("read_fact", "inspect_metadata", "python_numeric", "search_fact", "read_then_python")
_SEARCH_FACTS = (
    ("Find the official page title for example.com using public web evidence. Return only the title.", "Example Domain"),
    ("Find the chemical symbol for gold using public web evidence. Return only the symbol.", "Au"),
    ("Find the capital city of Japan using public web evidence. Return only the city.", "Tokyo"),
    ("Find the planet commonly called the Red Planet using public web evidence. Return only the planet name.", "Mars"),
)


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


def generate_policy_tasks(output_path: str | Path, *, count: int = 64, seed: str = "p4-policy-v1") -> Path:
    """Generate a small deterministic, non-GAIA policy curriculum.

    v2 makes tool choice less scripted than v1: prompts do not name the required
    tool, search is represented, and one category requires a two-tool flow.
    The task answers remain deterministic and independent of GAIA evaluation text.
    """
    if count < len(_CATEGORIES) or count > 128:
        raise ValueError(f"count must be between {len(_CATEGORIES)} and 128")
    rng = random.Random(seed)
    rows: list[dict] = []
    search_index = 0
    for index in range(count):
        category = _CATEGORIES[index % len(_CATEGORIES)]
        task_id = f"policy-{index + 1:03d}"
        if category == "read_fact":
            key = f"CODE_{index + 1:03d}"
            value = f"VALUE_{rng.randrange(1000, 9999)}"
            filename = f"facts-{index + 1:03d}.txt"
            rows.append(_base_row(
                task_id=task_id, category=category,
                question=f"In {filename}, what value is recorded for {key}? Return only the value.",
                expected=value, required_tools=["read"],
                files={filename: f"ALPHA=ignore\n{key}={value}\nOMEGA=ignore\n"},
            ))
        elif category == "inspect_metadata":
            filename = f"table-{index + 1:03d}.csv"
            rows.append(_base_row(
                task_id=task_id, category=category,
                question=f"How many columns does {filename} have? Return only the integer.",
                expected="3", required_tools=["inspect"],
                files={filename: "name,score,status\na,10,ok\nb,20,ok\n"},
            ))
        elif category == "python_numeric":
            a, b = rng.randrange(111, 800), rng.randrange(111, 800)
            expected = str(a * b)
            rows.append(_base_row(
                task_id=task_id, category=category,
                question=f"Compute {a} * {b} exactly. Return only the integer result.",
                expected=expected, required_tools=["python"],
            ))
        elif category == "search_fact":
            question, expected = _SEARCH_FACTS[search_index % len(_SEARCH_FACTS)]
            search_index += 1
            rows.append(_base_row(
                task_id=task_id, category=category,
                question=question, expected=expected, required_tools=["search"],
            ))
        else:
            a, b = rng.randrange(20, 90), rng.randrange(20, 90)
            filename = f"numbers-{index + 1:03d}.txt"
            rows.append(_base_row(
                task_id=task_id, category=category,
                question=(f"{filename} contains two integers. Return their exact product and nothing else."),
                expected=str(a * b), required_tools=["read", "python"],
                files={filename: f"left={a}\nright={b}\n"},
            ))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite different policy task set: {output_path}")
    output_path.write_text(text, encoding="utf-8")
    return output_path
