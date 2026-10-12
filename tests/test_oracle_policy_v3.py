from __future__ import annotations

from collections import Counter
import json
import re


CAPABILITIES = {
    "tool_selection",
    "argument_grounding",
    "multi_step",
    "evidence_to_final",
    "no_tool_stop",
    "failure_recovery",
    "strategy_switch",
    "duplicate_avoidance",
}


def _load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_policy_v3_has_frozen_balanced_splits_and_distinct_prompt_families(tmp_path):
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset

    paths = generate_oracle_policy_v3_dataset(tmp_path / "oracle", seed="small-agent-policy-v3")
    expected_counts = {"train": 1600, "dev": 200, "test": 200}

    families = {}
    for split, expected_count in expected_counts.items():
        rows = _load(paths[split])
        assert len(rows) == expected_count
        counts = Counter(row["capability"] for row in rows)
        assert set(counts) == CAPABILITIES
        assert set(counts.values()) == {expected_count // len(CAPABILITIES)}
        assert all(row["source"] == f"synthetic-small-agent-policy-v3/{split}" for row in rows)
        families[split] = {row["prompt_family"] for row in rows}
        assert len(families[split]) == 1

    assert families["train"].isdisjoint(families["dev"])
    assert families["train"].isdisjoint(families["test"])
    assert families["dev"].isdisjoint(families["test"])


def test_policy_v3_does_not_put_the_policy_answer_in_user_wording(tmp_path):
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset

    paths = generate_oracle_policy_v3_dataset(tmp_path / "oracle", seed="small-agent-policy-v3")
    rows = _load(paths["train"])

    recovery_forbidden = (
        "if search fails",
        "if that search",
        "recover by",
        "switch strategy",
        "do not repeat",
        "search once",
    )
    for row in rows:
        user = next(message["content"] for message in row["messages"] if message.get("role") == "user")
        folded = user.casefold()
        if row["capability"] in {"failure_recovery", "strategy_switch", "duplicate_avoidance"}:
            assert not any(phrase in folded for phrase in recovery_forbidden)
        if row["capability"] in {"tool_selection", "argument_grounding"}:
            assert re.search(r"\b(search|read|inspect|python)\b", folded) is None


def test_policy_v3_recovery_is_conditioned_on_an_observed_failure(tmp_path):
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset

    paths = generate_oracle_policy_v3_dataset(tmp_path / "oracle", seed="small-agent-policy-v3")
    rows = _load(paths["train"])
    recovery = next(row for row in rows if row["capability"] == "failure_recovery")
    messages = recovery["messages"]

    error_positions = [
        i for i, message in enumerate(messages)
        if message.get("role") == "tool" and "ERROR[" in str(message.get("content") or "")
    ]
    assert error_positions
    error_index = error_positions[0]
    assert any(message.get("role") == "assistant" and message.get("tool_calls") for message in messages[error_index + 1 :])
    assert recovery["verification"]["expected_tool_failures_only"] is True
