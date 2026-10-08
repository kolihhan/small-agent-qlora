from gaia_small_agent.training.policy_tasks import generate_policy_tasks


def test_curriculum_covers_search_and_implicit_tool_choice(tmp_path):
    path = generate_policy_tasks(tmp_path / "tasks.jsonl", count=64, seed="p4-policy-v1")
    rows = [__import__("json").loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    required = {tool for row in rows for tool in row["required_tools"]}
    assert "search" in required

    implicit = [
        row for row in rows
        if row["required_tools"] and not any(tool in row["question"].casefold() for tool in row["required_tools"])
    ]
    assert implicit, "curriculum should contain tasks where the prompt does not name the required tool"

    assert any(len(row["required_tools"]) >= 2 for row in rows), "curriculum should include multi-step tool flows"
