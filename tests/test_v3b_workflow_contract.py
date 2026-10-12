from pathlib import Path


WORKFLOW = Path(".github/workflows/hf-zero-gpu-train-v3b.yml")


def test_v3b_zero_gpu_workflow_freezes_diagnostic_training_contract():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Qwen/Qwen3.5-9B" in text
    assert 'EXPERIMENT_COMMIT: "ad8e72e9bdfdccda978e5538470b68fa08f4eccc"' in text
    assert 'TRAIN_STEPS: "64"' in text
    assert 'EXPOSURE_COUNT: "64"' in text
    assert 'EXPOSURE_SEED: "qlora-v3b-exposure"' in text
    assert 'EXPOSURE_PROFILE: "v3b-diagnostic"' in text
    assert 'MAX_LENGTH: "1024"' in text
    assert 'GRAD_ACC: "1"' in text

    assert "generate_oracle_policy_v3_dataset" in text
    assert "seed='small-agent-policy-v3'" in text
    assert "exposure_profile=os.environ['EXPOSURE_PROFILE']" in text
    assert "require_full_prompt=True" in text
    assert "max_length=int(os.environ['MAX_LENGTH'])" in text

    assert "'final': 29, 'tool_call': 35" in text
    assert "'evidence_to_final': 20" in text
    assert "'multi_step': 12" in text
    assert "'failure_recovery': 8" in text
    assert "training-exposure.json" in text
    assert "run_config.json" in text
    assert "minimum_safe_context" in text
    assert "qwen35-9b-qlora-v3b-adapter" in text


def test_v3b_zero_gpu_reservation_stays_inside_observed_free_quota():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "@spaces.GPU(duration=195)" in text
    assert "required_reservation_seconds': 293" in text
