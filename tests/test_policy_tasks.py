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
        assert row["source"] == "synthetic-policy-v2"
        assert row["license"] == "CC0-1.0"
        assert row["generator"] == "p4-policy-tasks"
        assert row["generator_version"] == "2"
        assert row["oracle_type"] == "exact"
        assert row["oracle_version"] == "1"
        assert isinstance(row["required_tools"], list)
        assert row["question"] and row["expected_answer"]
        # Curriculum should teach tool choice, not merely obey an explicit tool-name instruction.
        assert "use the python tool" not in row["question"].lower()
        assert "use the read tool" not in row["question"].lower()
        assert "use the inspect tool" not in row["question"].lower()


def test_policy_curriculum_contains_real_multi_step_tool_choice(tmp_path):
    from gaia_small_agent.training.policy_tasks import generate_policy_tasks

    path = tmp_path / "tasks.jsonl"
    generate_policy_tasks(path, count=64, seed="fixed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    multi = [row for row in rows if row["category"] == "read_then_python"]
    assert len(multi) == 16
    assert all(row["required_tools"] == ["read", "python"] for row in multi)
    assert all(row["files"] for row in multi)
