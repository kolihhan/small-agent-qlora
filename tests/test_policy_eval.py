import json

from gaia_small_agent.agent.types import AgentResult, RunMetrics, TraceEvent


def _row(task_id, question, expected, *, required_tools=None, files=None, fixtures=None, capability="tool_selection"):
    return {
        "task_id": task_id,
        "eval_task": {
            "question": question,
            "expected_answer": expected,
            "required_tools": required_tools or [],
            "files": files or {},
            "search_fixtures": fixtures or {},
            "capability": capability,
        },
    }


def test_policy_eval_aggregates_exact_completion_tools_and_stop_reasons(tmp_path):
    from gaia_small_agent.training.policy_eval import evaluate_policy_tasks

    tasks = tmp_path / "test.jsonl"
    rows = [
        _row("a", "q1", "42", required_tools=["python"]),
        _row("b", "q2", "yes", capability="no_tool_stop"),
        _row("c", "q3", "x"),
    ]
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    results = [
        AgentResult(
            "42",
            True,
            "final",
            [TraceEvent("tool_call", 1, {"id": "1", "name": "python", "arguments": {}})],
            RunMetrics(steps=2, tool_calls=1, tool_successes=1),
        ),
        AgentResult("no", True, "final", [], RunMetrics(steps=1)),
        AgentResult("", False, "model_timeout", [], RunMetrics(steps=1)),
    ]

    class Runtime:
        def __init__(self, result):
            self.result = result
            self.tools = {}

        def run(self, question, workspace):
            return self.result

    queue = list(results)
    summary = evaluate_policy_tasks(tasks, lambda: Runtime(queue.pop(0)), tmp_path / "work")

    assert summary["total"] == 3
    assert summary["correct"] == 1
    assert summary["completed"] == 2
    assert summary["exact_success_rate"] == 100 / 3
    assert summary["completion_rate"] == 200 / 3
    assert summary["required_tools_satisfied"] == 3
    assert summary["tool_calls"] == 1
    assert summary["tool_errors"] == 0
    assert summary["duplicate_calls_blocked"] == 0
    assert summary["stop_reasons"] == {"final": 2, "model_timeout": 1}
    assert summary["by_capability"]["tool_selection"]["correct"] == 1
    assert summary["by_capability"]["no_tool_stop"]["correct"] == 0


def test_policy_eval_materializes_files_before_runtime(tmp_path):
    from gaia_small_agent.training.policy_eval import evaluate_policy_tasks

    tasks = tmp_path / "test.jsonl"
    row = _row("file-task", "read it", "SECRET", files={"fact.txt": "SECRET\n"}, required_tools=["read"])
    tasks.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class Runtime:
        tools = {}

        def run(self, question, workspace):
            assert (workspace / "fact.txt").read_text(encoding="utf-8") == "SECRET\n"
            return AgentResult(
                "SECRET",
                True,
                "final",
                [TraceEvent("tool_call", 1, {"id": "1", "name": "read", "arguments": {"source": "fact.txt"}})],
                RunMetrics(steps=2, tool_calls=1, tool_successes=1),
            )

    summary = evaluate_policy_tasks(tasks, Runtime, tmp_path / "work")
    assert summary["correct"] == 1


def test_policy_eval_replaces_search_with_deterministic_fixture_backend(tmp_path):
    from gaia_small_agent.training.policy_eval import evaluate_policy_tasks
    from gaia_small_agent.tools.base import ToolResult

    tasks = tmp_path / "test.jsonl"
    observation = "Release code is CODE-777."
    row = _row(
        "search-task",
        "find PROJECT-777 release code",
        "CODE-777",
        required_tools=["search"],
        fixtures={"PROJECT-777": observation},
    )
    tasks.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class BrokenSearch:
        name = "search"
        description = "broken"
        schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

        def definition(self):
            return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.schema}}

        def run(self, arguments, workspace):
            return ToolResult(False, "live search must not be used", "LIVE_SEARCH")

    class Runtime:
        def __init__(self):
            self.tools = {"search": BrokenSearch()}

        def run(self, question, workspace):
            result = self.tools["search"].run({"query": "PROJECT-777 release code"}, workspace)
            assert result.ok is True
            assert result.content == observation
            return AgentResult(
                "CODE-777",
                True,
                "final",
                [TraceEvent("tool_call", 1, {"id": "1", "name": "search", "arguments": {"query": "PROJECT-777 release code"}})],
                RunMetrics(steps=2, tool_calls=1, tool_successes=1),
            )

    summary = evaluate_policy_tasks(tasks, Runtime, tmp_path / "work")
    assert summary["correct"] == 1
