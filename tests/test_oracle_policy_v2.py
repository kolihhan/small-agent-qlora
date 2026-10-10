import json


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_oracle_policy_v2_is_balanced_gaia_independent_and_deterministic(tmp_path):
    from gaia_small_agent.training.oracle_policy_v2 import generate_oracle_policy_v2_dataset

    a = generate_oracle_policy_v2_dataset(tmp_path / "a", seed="fixed-v2")
    b = generate_oracle_policy_v2_dataset(tmp_path / "b", seed="fixed-v2")
    expected_counts = {"train": 1600, "dev": 200, "test": 200}
    capabilities = {
        "tool_selection",
        "argument_grounding",
        "multi_step",
        "evidence_to_final",
        "no_tool_stop",
        "failure_recovery",
        "strategy_switch",
        "duplicate_avoidance",
    }

    for split, count in expected_counts.items():
        assert a[split].read_bytes() == b[split].read_bytes()
        rows = _rows(a[split])
        assert len(rows) == count
        assert {row["capability"] for row in rows} == capabilities
        assert all("gaia" not in row["source"].casefold() for row in rows)
        assert all(row["provenance"]["generator_version"] == "2" for row in rows)
        for capability in capabilities:
            assert sum(row["capability"] == capability for row in rows) == count // len(capabilities)


def test_v2_recovery_rows_encode_expected_failures_and_are_trainable(tmp_path):
    from gaia_small_agent.training.oracle_policy_v2 import generate_oracle_policy_v2_dataset
    from gaia_small_agent.training.qlora import load_verified_rows

    paths = generate_oracle_policy_v2_dataset(tmp_path / "v2", seed="recovery")
    rows = _rows(paths["train"])
    recovery = [row for row in rows if row["capability"] in {"failure_recovery", "strategy_switch"}]
    assert recovery
    assert all(row["verification"]["zero_tool_errors"] is False for row in recovery)
    assert all(row["verification"]["expected_tool_failures_only"] is True for row in recovery)
    assert all(row["metrics"]["tool_errors"] == 1 for row in recovery)
    assert all(any("ERROR[" in str(message.get("content", "")) for message in row["messages"] if message.get("role") == "tool") for row in recovery)

    loaded = load_verified_rows(paths["train"])
    assert len(loaded) == 1600


def test_v2_duplicate_avoidance_stops_after_evidence(tmp_path):
    from gaia_small_agent.training.oracle_policy_v2 import generate_oracle_policy_v2_dataset

    paths = generate_oracle_policy_v2_dataset(tmp_path / "v2", seed="duplicates")
    rows = _rows(paths["dev"])
    duplicate_rows = [row for row in rows if row["capability"] == "duplicate_avoidance"]
    assert duplicate_rows
    for row in duplicate_rows:
        calls = [
            call
            for message in row["messages"]
            if message.get("role") == "assistant"
            for call in (message.get("tool_calls") or [])
        ]
        assert len(calls) == 1
        assert row["messages"][-1] == {"role": "assistant", "content": row["expected_answer"]}
