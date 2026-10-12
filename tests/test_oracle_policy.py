import json


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_oracle_policy_dataset_is_balanced_deterministic_and_disjoint(tmp_path):
    from gaia_small_agent.training.oracle_policy import generate_oracle_policy_dataset

    first = tmp_path / "first"
    second = tmp_path / "second"
    a = generate_oracle_policy_dataset(first, seed="fixed")
    b = generate_oracle_policy_dataset(second, seed="fixed")

    expected_counts = {"train": 1600, "dev": 200, "test": 200}
    categories = {
        "tool_selection",
        "argument_grounding",
        "multi_step",
        "evidence_to_final",
        "no_tool_stop",
    }
    seen_ids = set()
    for split, count in expected_counts.items():
        assert a[split].read_bytes() == b[split].read_bytes()
        rows = _rows(a[split])
        assert len(rows) == count
        assert {row["capability"] for row in rows} == categories
        per_category = count // len(categories)
        for category in categories:
            assert sum(row["capability"] == category for row in rows) == per_category
        ids = {row["task_id"] for row in rows}
        assert len(ids) == count
        assert seen_ids.isdisjoint(ids)
        seen_ids |= ids


def test_oracle_rows_match_verified_training_contract_and_expose_all_tools(tmp_path):
    from gaia_small_agent.training.oracle_policy import generate_oracle_policy_dataset
    from gaia_small_agent.training.qlora import load_verified_rows

    paths = generate_oracle_policy_dataset(tmp_path / "data", seed="contract")
    rows = load_verified_rows(paths["train"])
    assert len(rows) == 1600

    expected_tools = {"search", "read", "inspect", "python"}
    for row in rows:
        assert row["verified"] is True
        assert "gaia" not in row["source"].casefold()
        assert row["split"] == "train"
        assert row["provenance"]["license"] == "CC0-1.0"
        assert row["provenance"]["generator"] == "tinyagent-policy-oracle"
        assert row["provenance"]["generator_version"] == "1"
        assert row["provenance"]["oracle_type"] == "deterministic-program"
        assert row["provenance"]["oracle_version"] == "1"
        assert {tool["function"]["name"] for tool in row["tools"]} == expected_tools
        assert row["messages"][0]["role"] == "system"
        assert row["messages"][1]["role"] == "user"
        assert row["eval_task"]["question"] == row["messages"][1]["content"]
        assert row["eval_task"]["expected_answer"] == row["expected_answer"]


def test_oracle_tool_calls_use_native_visible_message_shape(tmp_path):
    from gaia_small_agent.training.oracle_policy import generate_oracle_policy_dataset

    paths = generate_oracle_policy_dataset(tmp_path / "data", seed="shape")
    rows = _rows(paths["dev"])
    tool_rows = [row for row in rows if row["capability"] != "no_tool_stop"]
    assert tool_rows

    for row in tool_rows:
        assistant_calls = [
            message
            for message in row["messages"]
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        assert assistant_calls
        for message in assistant_calls:
            assert message.get("content") == ""
            for call in message["tool_calls"]:
                assert call["type"] == "function"
                assert call["id"]
                assert call["function"]["name"] in {"search", "read", "inspect", "python"}
                assert isinstance(call["function"]["arguments"], dict)

        tool_messages = [message for message in row["messages"] if message.get("role") == "tool"]
        assert tool_messages
        call_ids = {
            call["id"]
            for message in assistant_calls
            for call in message["tool_calls"]
        }
        assert all(message["tool_call_id"] in call_ids for message in tool_messages)


def test_no_tool_stop_rows_have_no_tool_messages_or_calls(tmp_path):
    from gaia_small_agent.training.oracle_policy import generate_oracle_policy_dataset

    paths = generate_oracle_policy_dataset(tmp_path / "data", seed="stop")
    rows = _rows(paths["test"])
    stop_rows = [row for row in rows if row["capability"] == "no_tool_stop"]
    assert len(stop_rows) == 40
    for row in stop_rows:
        assert row["required_tools"] == []
        assert all(message.get("role") != "tool" for message in row["messages"])
        assert all(not message.get("tool_calls") for message in row["messages"])
        assert row["messages"][-1] == {"role": "assistant", "content": row["expected_answer"]}
