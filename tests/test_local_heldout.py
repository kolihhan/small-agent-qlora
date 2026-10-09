import json
from pathlib import Path

from gaia_small_agent.agent.types import AgentResult, RunMetrics, TraceEvent
from gaia_small_agent.benchmark.local_heldout import load_cases, run_heldout
from gaia_small_agent.training.policy_tasks import generate_policy_tasks


BENCHMARK_PATH = Path(__file__).parents[1] / "evaluation" / "local-heldout-v1" / "cases.jsonl"


def _result(answer, *, tools=(), completed=True):
    trace = []
    for index, name in enumerate(tools, 1):
        trace.append(TraceEvent("tool_call", index, {"id": f"call-{index}", "name": name, "arguments": {}}))
        trace.append(TraceEvent("tool_result", index, {"id": f"call-{index}", "name": name, "ok": True, "error_code": None, "content": "ok"}))
    if completed:
        trace.append(TraceEvent("final", max(1, len(tools) + 1), {"answer": answer}))
    return AgentResult(
        answer=answer,
        completed=completed,
        stop_reason="final" if completed else "max_steps",
        trace=trace,
        metrics=RunMetrics(
            steps=max(1, len(tools) + 1),
            tool_calls=len(tools),
            tool_successes=len(tools),
        ),
    )


def test_frozen_heldout_is_balanced_unique_and_separate_from_training(tmp_path):
    cases = load_cases(BENCHMARK_PATH)

    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == 12
    assert {case["category"] for case in cases} == {"read", "inspect", "compute", "multi_step"}
    assert all(sum(case["category"] == category for case in cases) == 3 for category in {"read", "inspect", "compute", "multi_step"})
    assert all(case["source"] == "local-heldout-v1" for case in cases)
    assert all(case["license"] == "CC0-1.0" for case in cases)

    training_path = generate_policy_tasks(tmp_path / "policy.jsonl", count=64)
    training_questions = {
        json.loads(line)["question"]
        for line in training_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert not ({case["question"] for case in cases} & training_questions)


def test_run_heldout_scores_answer_and_required_tool_policy_separately(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    rows = [
        {
            "id": "heldout-a",
            "category": "read",
            "source": "local-heldout-v1",
            "license": "CC0-1.0",
            "question": "Return the code from note.txt.",
            "expected_answer": "42",
            "required_tools": ["read"],
            "files": {"note.txt": "code=42\n"},
        },
        {
            "id": "heldout-b",
            "category": "compute",
            "source": "local-heldout-v1",
            "license": "CC0-1.0",
            "question": "Return the computed integer.",
            "expected_answer": "99",
            "required_tools": ["python"],
            "files": {},
        },
        {
            "id": "heldout-c",
            "category": "compute",
            "source": "local-heldout-v1",
            "license": "CC0-1.0",
            "question": "Return the computed integer.",
            "expected_answer": "7",
            "required_tools": ["python"],
            "files": {},
        },
    ]
    cases_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cases = load_cases(cases_path)

    scripted = iter([
        _result("42", tools=("read",)),
        _result("98", tools=("python",)),
        _result("7", tools=()),
    ])

    class FakeRuntime:
        def __init__(self, result):
            self.result = result

        def run(self, question, workspace):
            if question == "Return the code from note.txt.":
                assert (Path(workspace) / "note.txt").read_text(encoding="utf-8") == "code=42\n"
            return self.result

    report = run_heldout(
        cases,
        runtime_factory=lambda: FakeRuntime(next(scripted)),
        work_root=tmp_path / "work",
        model={"name": "fake", "digest": "sha256:test"},
    )

    assert report["schema_version"] == "small-agent-heldout/v1"
    assert report["sample_size"] == 3
    assert report["metrics"]["accuracy"] == 2 / 3
    assert report["metrics"]["required_tool_pass_rate"] == 2 / 3
    assert report["metrics"]["full_pass_rate"] == 1 / 3
    assert report["cases"][0]["correct"] is True
    assert report["cases"][0]["required_tools_satisfied"] is True
    assert report["cases"][0]["full_pass"] is True
    assert report["cases"][1]["correct"] is False
    assert report["cases"][1]["required_tools_satisfied"] is True
    assert report["cases"][1]["full_pass"] is False
    assert report["cases"][2]["correct"] is True
    assert report["cases"][2]["required_tools_satisfied"] is False
    assert report["cases"][2]["full_pass"] is False
