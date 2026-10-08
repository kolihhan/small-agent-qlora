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
    assert {row["category"] for row in rows} == {
        "read_fact", "inspect_metadata", "python_numeric", "read_then_python"
    }
    for row in rows:
        assert "gaia" not in row["source"].lower()
        assert row["license"] == "CC0-1.0"
        assert row["generator"] == "p4-policy-tasks"
        assert row["generator_version"] == "2"
        assert row["oracle_type"] == "exact"
        assert row["oracle_version"] == "1"
        assert isinstance(row["required_tools"], list)
        assert row["question"] and row["expected_answer"]
        # The policy must infer the tool choice; prompts must not name tools.
        lowered = row["question"].lower()
        assert "python tool" not in lowered
        assert "read tool" not in lowered
        assert "inspect tool" not in lowered


def test_read_then_python_requires_a_real_multi_step_policy(tmp_path):
    from gaia_small_agent.training.policy_tasks import generate_policy_tasks

    path = tmp_path / "tasks.jsonl"
    generate_policy_tasks(path, count=4, seed="fixed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    row = next(item for item in rows if item["category"] == "read_then_python")
    assert row["required_tools"] == ["read", "python"]
    assert len(row["files"]) == 1
    assert "product" in row["question"].lower()
