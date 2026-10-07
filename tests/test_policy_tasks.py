import json


def test_policy_task_generator_is_deterministic_independent_and_provenance_complete(tmp_path):
    from gaia_small_agent.training.policy_tasks import generate_policy_tasks

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    generate_policy_tasks(a, count=64, seed="fixed")
    generate_policy_tasks(b, count=64, seed="fixed")
    assert a.read_bytes() == b.read_bytes()
    rows = [json.loads(line) for line in a.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 64
    assert {row["category"] for row in rows} == {"read_fact", "inspect_metadata", "python_numeric", "format_recovery"}
    for row in rows:
        assert "gaia" not in row["source"].lower()
        assert row["license"] == "CC0-1.0"
        assert row["generator"] == "p4-policy-tasks"
        assert row["generator_version"] == "2"
        assert row["oracle_type"] == "exact"
        assert row["oracle_version"] == "1"
        assert isinstance(row["required_tools"], list)
        assert row["question"] and row["expected_answer"]


def test_policy_prompts_do_not_tell_the_model_which_required_tool_to_call(tmp_path):
    from gaia_small_agent.training.policy_tasks import generate_policy_tasks

    output = generate_policy_tasks(tmp_path / "policy.jsonl", count=64, seed="fixed")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    for row in rows:
        question = row["question"].casefold()
        for tool_name in row["required_tools"]:
            assert tool_name.casefold() not in question
