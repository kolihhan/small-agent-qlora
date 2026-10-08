import json


def test_policy_v2_requires_tool_selection_and_multistep_local_work(tmp_path):
    from gaia_small_agent.training.policy_tasks import generate_policy_tasks

    output = tmp_path / "policy.jsonl"
    generate_policy_tasks(output, count=64, seed="p4-policy-v2")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert {row["category"] for row in rows} == {
        "read_fact",
        "inspect_metadata",
        "read_then_python",
        "inspect_then_read",
    }
    assert all(row["generator_version"] == "2" for row in rows)
    assert any(row["required_tools"] == ["read", "python"] for row in rows)
    assert any(row["required_tools"] == ["inspect", "read"] for row in rows)

    # The curriculum should train tool choice rather than parroting an explicit
    # instruction naming the expected tool.
    for row in rows:
        question = row["question"].casefold()
        for tool_name in row["required_tools"]:
            assert f"use the {tool_name} tool" not in question
