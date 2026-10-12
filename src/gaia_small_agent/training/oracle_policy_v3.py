from __future__ import annotations

import json
import random
from pathlib import Path

from .oracle_policy import _base_row, _call, _tool_result

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
_PROMPT_FAMILIES = {
    "train": "contextual-train-a",
    "dev": "contextual-dev-b",
    "test": "contextual-test-c",
}
_PROVENANCE = {
    "license": "CC0-1.0",
    "generator": "small-agent-policy-oracle",
    "generator_version": "3",
    "oracle_type": "deterministic-program",
    "oracle_version": "3",
}


def _phrasing(split: str, *, train: str, dev: str, test: str) -> str:
    return {"train": train, "dev": dev, "test": test}[split]


def _upgrade(row: dict, split: str) -> dict:
    row["source"] = f"synthetic-small-agent-policy-v3/{split}"
    row["provenance"] = dict(_PROVENANCE)
    row["prompt_family"] = _PROMPT_FAMILIES[split]
    return row


def _mark_expected_failure(row: dict, error_code: str) -> dict:
    row["verification"]["zero_tool_errors"] = False
    row["verification"]["expected_tool_failures_only"] = True
    row["verification"]["expected_tool_error_codes"] = [error_code]
    row["metrics"]["tool_errors"] = 1
    row["metrics"]["tool_successes"] = max(0, row["metrics"]["tool_calls"] - 1)
    return row


def _tool_selection_case(rng: random.Random, task_id: str, split: str, index: int) -> dict:
    kind = ("python", "read", "inspect", "search")[index % 4]
    call_id = f"{task_id}-call-1"

    if kind == "python":
        a, b = rng.randrange(31, 900), rng.randrange(13, 120)
        expected = str(a * b)
        question = _phrasing(
            split,
            train=f"Give the exact product of {a} and {b}. Return only the integer.",
            dev=f"Resolve {a} × {b} exactly and reply with the number alone.",
            test=f"Only the exact integer is needed: {a} multiplied by {b}.",
        )
        messages = [_call(call_id, "python", {"code": f"print({a} * {b})"}), _tool_result(call_id, "python", expected), {"role": "assistant", "content": expected}]
        return _upgrade(_base_row(task_id=task_id, split=split, capability="tool_selection", question=question, expected=expected, required_tools=["python"], messages=messages), split)

    if kind == "read":
        key = f"FIELD_{rng.randrange(100000, 999999)}"
        value = f"VALUE_{rng.randrange(100000, 999999)}"
        filename = f"workspace-note-{rng.randrange(100000, 999999)}.txt"
        content = f"owner=ops\n{key}={value}\nstatus=current\n"
        question = _phrasing(
            split,
            train=f"In workspace note {filename}, what value is stored for {key}? Return only the value.",
            dev=f"The artifact {filename} contains a field named {key}. Give its value only.",
            test=f"Resolve {key} from the workspace artifact {filename}; answer with the value alone.",
        )
        messages = [_call(call_id, "read", {"source": filename}), _tool_result(call_id, "read", content), {"role": "assistant", "content": value}]
        return _upgrade(_base_row(task_id=task_id, split=split, capability="tool_selection", question=question, expected=value, required_tools=["read"], messages=messages, files={filename: content}), split)

    if kind == "inspect":
        filename = f"table-{rng.randrange(100000, 999999)}.csv"
        columns = rng.randrange(3, 8)
        header = ",".join(f"field_{i}" for i in range(1, columns + 1))
        content = f"{header}\n" + ",".join(str(rng.randrange(1, 99)) for _ in range(columns)) + "\n"
        observation = f"CSV metadata: rows=1, columns={columns}, headers=[{header}]"
        expected = str(columns)
        question = _phrasing(
            split,
            train=f"How many columns are in workspace artifact {filename}? Return only the integer.",
            dev=f"For the tabular artifact {filename}, give the column count only.",
            test=f"Determine the number of fields in {filename}; answer with the integer alone.",
        )
        messages = [_call(call_id, "inspect", {"path": filename}), _tool_result(call_id, "inspect", observation), {"role": "assistant", "content": expected}]
        return _upgrade(_base_row(task_id=task_id, split=split, capability="tool_selection", question=question, expected=expected, required_tools=["inspect"], messages=messages, files={filename: content}), split)

    project = f"PROJECT-{rng.randrange(100000, 999999)}"
    code = f"CODE-{rng.randrange(100000, 999999)}"
    observation = f"[1] Public release bulletin\nURL: https://example.invalid/{project}\nCurrent release code for {project}: {code}."
    question = _phrasing(
        split,
        train=f"What current release code is publicly associated with {project}? Return only the code.",
        dev=f"Give the public release identifier currently listed for {project}; code only.",
        test=f"Resolve the current public release code for {project} and answer with that code alone.",
    )
    messages = [_call(call_id, "search", {"query": f"{project} current release code", "max_results": 5}), _tool_result(call_id, "search", observation), {"role": "assistant", "content": code}]
    return _upgrade(_base_row(task_id=task_id, split=split, capability="tool_selection", question=question, expected=code, required_tools=["search"], messages=messages, search_fixtures={project: observation}), split)


def _argument_grounding_case(rng: random.Random, task_id: str, split: str, index: int) -> dict:
    kind = ("read", "inspect", "search", "python")[index % 4]
    call_id = f"{task_id}-call-1"

    if kind == "read":
        target = f"ledger-{rng.randrange(100000, 999999)}.txt"
        distractor = f"ledger-{rng.randrange(100000, 999999)}.txt"
        key = f"CURRENT_{rng.randrange(1000, 9999)}"
        value = f"TOKEN_{rng.randrange(100000, 999999)}"
        files = {target: f"{key}={value}\n", distractor: f"{key}=OLD_VALUE\n"}
        question = _phrasing(
            split,
            train=f"The current ledger is {target}, not {distractor}. Return the value for {key} only.",
            dev=f"Use artifact {target} as the authoritative ledger. What is {key}? Value only.",
            test=f"Authoritative source: {target}. Resolve {key} without using the similarly named {distractor}.",
        )
        messages = [_call(call_id, "read", {"source": target}), _tool_result(call_id, "read", files[target]), {"role": "assistant", "content": value}]
        return _upgrade(_base_row(task_id=task_id, split=split, capability="argument_grounding", question=question, expected=value, required_tools=["read"], messages=messages, files=files), split)

    if kind == "inspect":
        target = f"report-{rng.randrange(100000, 999999)}.csv"
        distractor = f"archive-{rng.randrange(100000, 999999)}.csv"
        columns = rng.randrange(4, 9)
        header = ",".join(f"c{i}" for i in range(columns))
        files = {target: f"{header}\n" + ",".join("1" for _ in range(columns)) + "\n", distractor: "a,b\n1,2\n"}
        observation = f"CSV metadata: rows=1, columns={columns}, headers=[{header}]"
        expected = str(columns)
        question = _phrasing(
            split,
            train=f"For the current report {target} rather than {distractor}, return its column count.",
            dev=f"Artifact {target} is the active report. Give only its number of columns.",
            test=f"Resolve the field count for active artifact {target}; {distractor} is historical.",
        )
        messages = [_call(call_id, "inspect", {"path": target}), _tool_result(call_id, "inspect", observation), {"role": "assistant", "content": expected}]
        return _upgrade(_base_row(task_id=task_id, split=split, capability="argument_grounding", question=question, expected=expected, required_tools=["inspect"], messages=messages, files=files), split)

    if kind == "search":
        project = f"PRODUCT-{rng.randrange(100000, 999999)}"
        region = rng.choice(("APAC", "EMEA", "LATAM"))
        code = f"{region}-{rng.randrange(10000, 99999)}"
        observation = f"[1] {project} {region} bulletin\nURL: https://example.invalid/{project}/{region}\nCurrent identifier: {code}."
        question = _phrasing(
            split,
            train=f"What is the current {region} identifier for {project}? Return only the identifier.",
            dev=f"Resolve {project}'s active identifier in region {region}; identifier only.",
            test=f"For {project}, give the current regional code for {region} and nothing else.",
        )
        messages = [_call(call_id, "search", {"query": f"{project} {region} current identifier", "max_results": 5}), _tool_result(call_id, "search", observation), {"role": "assistant", "content": code}]
        return _upgrade(_base_row(task_id=task_id, split=split, capability="argument_grounding", question=question, expected=code, required_tools=["search"], messages=messages, search_fixtures={f"{project} {region}": observation}), split)

    a, b, c = rng.randrange(10, 90), rng.randrange(10, 90), rng.randrange(2, 9)
    expected = str((a + b) * c)
    question = _phrasing(
        split,
        train=f"Return only the exact result of ({a} + {b}) × {c}.",
        dev=f"Compute exactly: add {a} and {b}, then multiply the sum by {c}. Number only.",
        test=f"Exact integer only for ({a}+{b})*{c}.",
    )
    messages = [_call(call_id, "python", {"code": f"print(({a} + {b}) * {c})"}), _tool_result(call_id, "python", expected), {"role": "assistant", "content": expected}]
    return _upgrade(_base_row(task_id=task_id, split=split, capability="argument_grounding", question=question, expected=expected, required_tools=["python"], messages=messages), split)


def _multi_step_case(rng: random.Random, task_id: str, split: str) -> dict:
    left, right = rng.randrange(20, 500), rng.randrange(20, 500)
    filename = f"metrics-{rng.randrange(100000, 999999)}.txt"
    content = f"baseline={left}\nadjustment={right}\nstatus=current\n"
    expected = str(left + right)
    question = _phrasing(
        split,
        train=f"Workspace artifact {filename} contains baseline and adjustment. Return their exact sum only.",
        dev=f"From artifact {filename}, combine the current baseline and adjustment into one exact total; number only.",
        test=f"Resolve the exact total of the two current numeric fields in {filename}. Return only that total.",
    )
    first_id, second_id = f"{task_id}-call-1", f"{task_id}-call-2"
    messages = [
        _call(first_id, "read", {"source": filename}),
        _tool_result(first_id, "read", content),
        _call(second_id, "python", {"code": f"print({left} + {right})"}),
        _tool_result(second_id, "python", expected),
        {"role": "assistant", "content": expected},
    ]
    return _upgrade(_base_row(task_id=task_id, split=split, capability="multi_step", question=question, expected=expected, required_tools=["read", "python"], messages=messages, files={filename: content}), split)


def _evidence_to_final_case(rng: random.Random, task_id: str, split: str, index: int) -> dict:
    project = f"RELEASE-{rng.randrange(100000, 999999)}"
    old_day = rng.randrange(1, 20)
    new_day = old_day + rng.randrange(2, 8)
    month = rng.choice(("June", "July", "August", "September"))
    year = rng.choice((2025, 2026))
    expected = f"{new_day} {month} {year}"
    noisy = (
        f"Draft planning note: {project} was initially listed for {old_day} {month} {year}.\n"
        "That draft is superseded.\n"
        f"Current release bulletin: the revised launch date for {project} is {expected}.\n"
        "Status: approved and current."
    )
    call_id = f"{task_id}-call-1"
    if index % 2 == 0:
        filename = f"release-status-{rng.randrange(100000, 999999)}.txt"
        question = _phrasing(
            split,
            train=f"What is the current approved launch date for {project} according to workspace artifact {filename}? Date only.",
            dev=f"Artifact {filename} contains both superseded and current schedule information for {project}. Give the current date only.",
            test=f"Resolve {project}'s approved current launch date from {filename}; ignore superseded schedule text.",
        )
        messages = [_call(call_id, "read", {"source": filename}), _tool_result(call_id, "read", noisy), {"role": "assistant", "content": expected}]
        return _upgrade(_base_row(task_id=task_id, split=split, capability="evidence_to_final", question=question, expected=expected, required_tools=["read"], messages=messages, files={filename: noisy}), split)

    question = _phrasing(
        split,
        train=f"What is the currently approved launch date for {project}? Return the date only.",
        dev=f"Resolve the active launch date for {project}; ignore superseded public schedule information.",
        test=f"Give {project}'s approved current launch date and nothing else.",
    )
    observation = f"[1] Status bulletin\nURL: https://example.invalid/{project}\n{noisy}"
    messages = [_call(call_id, "search", {"query": f"{project} current approved launch date", "max_results": 5}), _tool_result(call_id, "search", observation), {"role": "assistant", "content": expected}]
    return _upgrade(_base_row(task_id=task_id, split=split, capability="evidence_to_final", question=question, expected=expected, required_tools=["search"], messages=messages, search_fixtures={project: observation}), split)


def _no_tool_case(rng: random.Random, task_id: str, split: str) -> dict:
    value = f"DIRECT-{rng.randrange(100000, 999999)}"
    question = _phrasing(
        split,
        train=f"The answer token is {value}. Return only that token.",
        dev=f"Reply with this supplied token and nothing else: {value}",
        test=f"Exact response required: {value}",
    )
    return _upgrade(_base_row(task_id=task_id, split=split, capability="no_tool_stop", question=question, expected=value, required_tools=[], messages=[{"role": "assistant", "content": value}]), split)


def _failure_recovery_case(rng: random.Random, task_id: str, split: str) -> dict:
    project = f"PRIMARY-{rng.randrange(100000, 999999)}"
    alias = f"MIRROR-{rng.randrange(100000, 999999)}"
    code = f"CODE-{rng.randrange(100000, 999999)}"
    question = _phrasing(
        split,
        train=f"Find the current release code for {project}. A public mirror is also known as {alias}. Return only the current code.",
        dev=f"Resolve {project}'s current public release code. The same release is indexed under alias {alias}. Code only.",
        test=f"Give the current release code for {project}; public records may also use alias {alias}.",
    )
    first_id, second_id = f"{task_id}-call-1", f"{task_id}-call-2"
    observation = f"[1] Mirror release bulletin\nURL: https://example.invalid/{alias}\nCurrent code for {project}: {code}."
    row = _base_row(
        task_id=task_id,
        split=split,
        capability="failure_recovery",
        question=question,
        expected=code,
        required_tools=["search"],
        messages=[
            _call(first_id, "search", {"query": f"{project} current release code", "max_results": 5}),
            _tool_result(first_id, "search", "ERROR[SEARCH_FIXTURE_MISS]: No deterministic search fixture matched query"),
            _call(second_id, "search", {"query": f"{alias} current release code", "max_results": 5}),
            _tool_result(second_id, "search", observation),
            {"role": "assistant", "content": code},
        ],
        search_fixtures={alias: observation},
    )
    return _mark_expected_failure(_upgrade(row, split), "SEARCH_FIXTURE_MISS")


def _strategy_switch_case(rng: random.Random, task_id: str, split: str) -> dict:
    project = f"PROJECT-{rng.randrange(100000, 999999)}"
    token = f"TOKEN-{rng.randrange(100000, 999999)}"
    filename = f"local-archive-{rng.randrange(100000, 999999)}.txt"
    content = f"Archive record\nproject={project}\ncurrent_token={token}\n"
    question = _phrasing(
        split,
        train=f"Find the current token for {project}. A local archive artifact named {filename} is available together with public references. Token only.",
        dev=f"Resolve {project}'s current token. Workspace artifact {filename} is an available local source in addition to public records.",
        test=f"Give the current token for {project}; the workspace includes archive artifact {filename}.",
    )
    first_id, second_id = f"{task_id}-call-1", f"{task_id}-call-2"
    row = _base_row(
        task_id=task_id,
        split=split,
        capability="strategy_switch",
        question=question,
        expected=token,
        required_tools=["search", "read"],
        messages=[
            _call(first_id, "search", {"query": f"{project} current token", "max_results": 5}),
            _tool_result(first_id, "search", "ERROR[SEARCH_FIXTURE_MISS]: No deterministic search fixture matched query"),
            _call(second_id, "read", {"source": filename}),
            _tool_result(second_id, "read", content),
            {"role": "assistant", "content": token},
        ],
        files={filename: content},
        search_fixtures={"NEVER-MATCH-THIS-V3-QUERY": "unused fixture"},
    )
    return _mark_expected_failure(_upgrade(row, split), "SEARCH_FIXTURE_MISS")


def _duplicate_avoidance_case(rng: random.Random, task_id: str, split: str) -> dict:
    project = f"CANON-{rng.randrange(100000, 999999)}"
    value = f"VALUE-{rng.randrange(100000, 999999)}"
    observation = f"[1] Canonical record\nURL: https://example.invalid/{project}\nCanonical current value: {value}."
    question = _phrasing(
        split,
        train=f"Find the canonical current value for {project}. Return only the value.",
        dev=f"What current canonical value is listed for {project}? Value only.",
        test=f"Resolve {project}'s canonical current value and answer with that value alone.",
    )
    call_id = f"{task_id}-call-1"
    messages = [_call(call_id, "search", {"query": f"{project} canonical current value", "max_results": 5}), _tool_result(call_id, "search", observation), {"role": "assistant", "content": value}]
    return _upgrade(_base_row(task_id=task_id, split=split, capability="duplicate_avoidance", question=question, expected=value, required_tools=["search"], messages=messages, search_fixtures={project: observation}), split)


def _row_for(capability: str, rng: random.Random, task_id: str, split: str, index: int) -> dict:
    if capability == "tool_selection":
        return _tool_selection_case(rng, task_id, split, index)
    if capability == "argument_grounding":
        return _argument_grounding_case(rng, task_id, split, index)
    if capability == "multi_step":
        return _multi_step_case(rng, task_id, split)
    if capability == "evidence_to_final":
        return _evidence_to_final_case(rng, task_id, split, index)
    if capability == "no_tool_stop":
        return _no_tool_case(rng, task_id, split)
    if capability == "failure_recovery":
        return _failure_recovery_case(rng, task_id, split)
    if capability == "strategy_switch":
        return _strategy_switch_case(rng, task_id, split)
    if capability == "duplicate_avoidance":
        return _duplicate_avoidance_case(rng, task_id, split)
    raise ValueError(f"unknown capability: {capability}")


def generate_oracle_policy_v3_dataset(
    output_dir: str | Path,
    *,
    seed: str = "small-agent-policy-v3",
) -> dict[str, Path]:
    """Generate deterministic GAIA-independent observation-conditioned policy trajectories."""
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
                task_id = f"small-agent-policy-v3-{split}-{capability}-{index + 1:04d}"
                rows.append(_row_for(capability, rng, task_id, split, index))

        path = output_dir / f"{split}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different oracle dataset: {path}")
        path.write_text(text, encoding="utf-8")
        paths[split] = path

    return paths


__all__ = ["generate_oracle_policy_v3_dataset"]
