from scripts.gaia_recovery_observation import oracle_error_messages


def test_oracle_serialization_changes_only_the_search_miss_observation():
    messages = [
        {"role": "user", "content": "Find the mirror."},
        {"role": "tool", "content": "ToolError[SEARCH_FIXTURE_MISS]: No deterministic search fixture matched query", "tool_call_id": "a"},
        {"role": "tool", "content": "ToolError[NOT_FOUND]: No file", "tool_call_id": "b"},
    ]
    changed = oracle_error_messages(messages)
    assert changed[0] == messages[0]
    assert changed[1] == {**messages[1], "content": "ERROR[SEARCH_FIXTURE_MISS]: No deterministic search fixture matched query"}
    assert changed[2] == messages[2]
    assert messages[1]["content"].startswith("ToolError")
