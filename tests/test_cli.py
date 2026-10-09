import json
from pathlib import Path

import pytest

from gaia_small_agent.agent.types import AgentResult, RunMetrics, TraceEvent
from gaia_small_agent.cli import build_parser, make_runtime


def test_run_parser_supports_transformers_adapter():
    args = build_parser().parse_args(["run", "hello", "--backend", "transformers", "--adapter", "adapters/demo"])
    assert args.backend == "transformers"
    assert args.adapter == "adapters/demo"
    assert args.hf_model == "Qwen/Qwen3.5-4B"
    assert args.cache_implementation == "default"


def test_make_runtime_can_use_transformers_default_cache(monkeypatch):
    import gaia_small_agent.cli as cli

    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cli, "TransformersQwenModel", FakeModel)
    args = build_parser().parse_args([
        "run", "hello", "--backend", "transformers", "--cache-implementation", "default",
    ])

    make_runtime(args)

    assert captured["cache_implementation"] is None


def test_collect_parser_exposes_verified_trajectory_pipeline():
    args = build_parser().parse_args([
        "collect-trajectories",
        "--tasks", "data/tasks.jsonl",
        "--output", "data/verified.jsonl",
    ])
    assert args.tasks == "data/tasks.jsonl"
    assert args.output == "data/verified.jsonl"
    assert args.backend == "ollama"


def test_make_runtime_forwards_max_new_tokens_to_ollama():
    args = build_parser().parse_args([
        "run", "hello", "--backend", "ollama", "--max-new-tokens", "128",
    ])

    runtime = make_runtime(args)

    assert runtime.model.max_new_tokens == 128


def test_make_runtime_forwards_thinking_to_ollama():
    default_args = build_parser().parse_args(["run", "hello", "--backend", "ollama"])
    thinking_args = build_parser().parse_args(["run", "hello", "--backend", "ollama", "--thinking"])

    assert make_runtime(default_args).model.enable_thinking is False
    assert make_runtime(thinking_args).model.enable_thinking is True


@pytest.mark.parametrize(("flag", "value"), [
    ("--max-steps", "0"),
    ("--max-steps", "-1"),
    ("--max-new-tokens", "0"),
    ("--max-new-tokens", "-1"),
])
def test_runtime_parser_rejects_non_positive_limits(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "hello", flag, value])


def test_doctor_parser_defaults_to_local_ollama():
    args = build_parser().parse_args(["doctor"])
    assert args.backend == "ollama"
    assert args.model == "qwen3.5:4b"
    assert args.ollama_url == "http://localhost:11434"
    assert args.workspace == "runs/doctor"


def _result():
    observation = "full observation " + ("x" * 220)
    return AgentResult(
        answer="done",
        completed=True,
        stop_reason="final",
        trace=[TraceEvent("tool_result", 1, {"name": "demo", "ok": True, "content": observation})],
        metrics=RunMetrics(steps=1, model_calls=1),
    )


def test_run_output_persists_full_result_json_and_runtime_provenance(monkeypatch, tmp_path, capsys):
    class FakeTool:
        def __init__(self, name):
            self.name = name

    class FakeRuntime:
        def __init__(self):
            self.tools = {name: FakeTool(name) for name in ("search", "read", "inspect", "python")}

        def run(self, question, workspace):
            return _result()

    monkeypatch.setattr("gaia_small_agent.cli.make_runtime", lambda args: FakeRuntime())
    output = tmp_path / "nested" / "result.json"
    args = build_parser().parse_args(["run", "hello", "--output", str(output)])

    assert args.func(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "small-agent-run/v1"
    assert payload["run_config"] == {
        "backend": "ollama",
        "model": "qwen3.5:4b",
        "adapter": None,
        "max_steps": 12,
        "max_new_tokens": 512,
        "thinking": False,
        "tool_observation_max_chars": 2000,
        "tool_names": ["search", "read", "inspect", "python"],
    }
    assert payload["answer"] == "done"
    assert payload["completed"] is True
    assert payload["stop_reason"] == "final"
    assert payload["metrics"] == {"steps": 1, "tool_calls": 0, "tool_successes": 0, "tool_errors": 0, "duplicate_calls_blocked": 0, "model_calls": 1}
    assert payload["trace"] == [{"kind": "tool_result", "step": 1, "data": {"name": "demo", "ok": True, "content": "full observation " + ("x" * 220)}}]
    assert "x" * 180 not in capsys.readouterr().out


def test_run_output_defaults_to_none():
    args = build_parser().parse_args(["run", "hello"])
    assert args.output is None
    assert args.workspace == "runs/task"


def test_gaia_command_passes_effective_run_config(monkeypatch, tmp_path):
    from gaia_small_agent.cli import gaia_command

    captured = {}
    monkeypatch.setattr("gaia_small_agent.cli.load_validation", lambda *args, **kwargs: [])
    monkeypatch.setattr("gaia_small_agent.cli.preflight_default_tools", lambda workspace: {"tool_names": ["search", "read", "inspect", "python"]})

    def fake_run_gaia100(dataset, factory, work_root, **kwargs):
        captured.update(kwargs)
        captured["work_root"] = work_root
        return {"total": 0}

    monkeypatch.setattr("gaia_small_agent.cli.run_gaia100", fake_run_gaia100)
    adapter = tmp_path / "adapter"
    args = build_parser().parse_args([
        "gaia100",
        "--backend", "transformers",
        "--hf-model", "synthetic/hf-model",
        "--adapter", str(adapter),
        "--no-4bit",
        "--thinking",
        "--max-steps", "7",
        "--max-new-tokens", "321",
        "--seed", "synthetic-seed",
        "--gaia-revision", "synthetic-revision",
        "--work-root", str(tmp_path / "run"),
        "--protected-questions", str(tmp_path / "protected.json"),
    ])

    assert gaia_command(args) == 0
    assert captured["run_config"] == {
        "backend": "transformers",
        "model": "synthetic/hf-model",
        "adapter": str(adapter.resolve()),
        "max_steps": 7,
        "max_new_tokens": 321,
        "thinking": True,
        "quantize_4bit": False,
        "seed": "synthetic-seed",
        "dataset_revision": "synthetic-revision",
        "tool_observation_max_chars": 2000,
        "partition": "diagnostic",
        "cache_implementation": "default",
        "tool_names": ["search", "read", "inspect", "python"],
    }
