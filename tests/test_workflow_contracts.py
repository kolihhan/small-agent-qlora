from pathlib import Path


WORKFLOW = Path(".github/workflows/gaia-shadow-gate.yml")
TRIGGER = Path(".github/GAIA_SHADOW_TRIGGER")
CPU_QLORA_WORKFLOW = Path(".github/workflows/qlora-cpu-batch.yml")
GAIA_9B_THINKING_WORKFLOW = Path(".github/workflows/gaia-9b-thinking20.yml")
GAIA_SHARD_RUNNER = Path("scripts/run_gaia_eval_shard.py")


def test_shadow_workflow_is_matched_and_aggregate_only():
    text = WORKFLOW.read_text(encoding="utf-8")

    credential_gate = text.index("Verify gated dataset credential")
    model_pull = text.index("Start Ollama and pull model")
    assert credential_gate < model_pull

    assert "baseline_sha=" in text
    assert "exposure=" in text
    assert "candidate_name=" in text
    assert "exposure must be 1, 2, or 3" in text
    assert "path: baseline" in text
    assert text.count("--partition shadow") == 2
    assert text.count("--max-steps 12") == 2
    assert text.count("--max-new-tokens 512") == 2
    assert "evaluate_shadow_promotion" in text

    assert "name: gaia-shadow-gate-summary" in text
    assert "path: runs/shadow-gate-summary.json" in text
    assert "path: runs/\n" not in text
    assert "path: runs/**" not in text
    assert "path: runs/*" not in text

    upload_block = text[text.index("Upload sanitized aggregate only"):]
    forbidden = (
        "results.jsonl",
        "manifest.json",
        "protected-question",
        "trace",
        "workspace",
        "gaia-baseline-shadow/",
        "gaia-candidate-shadow/",
    )
    for token in forbidden:
        assert token not in upload_block


def test_shadow_trigger_has_explicit_bounded_exposure_metadata():
    values = {}
    for line in TRIGGER.read_text(encoding="utf-8").splitlines():
        if line.strip() and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value

    assert set(values) == {"baseline_sha", "exposure", "candidate_name"}
    assert len(values["baseline_sha"]) == 40
    assert all(ch in "0123456789abcdef" for ch in values["baseline_sha"])
    assert values["exposure"] in {"1", "2", "3"}
    assert values["candidate_name"]


def test_cpu_qlora_batch_keeps_frozen_max_length():
    text = CPU_QLORA_WORKFLOW.read_text(encoding="utf-8")

    assert '--max-length 1536' in text
    assert '--max-length 1024' not in text


def test_cpu_qlora_batch_avoids_regressed_kernel_loader():
    text = CPU_QLORA_WORKFLOW.read_text(encoding="utf-8")

    assert 'python -m pip install "kernels>=0.11.1"' not in text


def test_gaia_9b_thinking_baseline_is_isolated_and_matched():
    workflow = GAIA_9B_THINKING_WORKFLOW.read_text(encoding="utf-8")
    runner = GAIA_SHARD_RUNNER.read_text(encoding="utf-8")

    assert 'parser.add_argument("--thinking", action="store_true")' in runner
    assert "enable_thinking=args.thinking" in runner
    assert '"thinking": args.thinking' in runner

    assert "MODEL: qwen3.5:9b" in workflow
    assert 'SHARD_COUNT: "20"' in workflow
    assert "max-parallel: 20" in workflow
    assert "--thinking" in workflow
    assert "--shard-count \"$SHARD_COUNT\"" in workflow
    assert "expected Evaluation100 total 100" in workflow
    assert "name: gaia-9b-thinking-baseline-summary-20way" in workflow
    assert "results.jsonl" not in workflow[workflow.index("Upload final sanitized aggregate only"):]
