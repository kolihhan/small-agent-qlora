from pathlib import Path


WORKFLOW = Path(".github/workflows/v3c-sft-heldout.yml")
APP = Path("scripts/v3c_space_app.py")


def test_v3c_workflow_freezes_training_and_eval_contract():
    text = WORKFLOW.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert 'TRAIN_STEPS: "128"' in text
    assert 'SELECTION_SEED: "qlora-v3c-full-trajectory"' in text
    assert 'MAX_LENGTH: "1024"' in text
    assert 'GRAD_ACC: "1"' in text
    assert "@spaces.GPU(duration=195)" in app
    assert "learning_rate=1e-4" in app
    assert "lora_r=16" in app
    assert "lora_alpha=32" in app
    assert "lora_dropout=0.05" in app
    assert 'lora_target_modules="all-linear"' in app
    assert "exposure_count=None" in app
    assert "require_full_prompt=True" in app

    assert 'SAMPLE_COUNT: "64"' in text
    assert "heldout-v3b-next-action-v1" in text
    assert "--reasoning-budget 256" in text
    assert "--temperature 0" in text
    assert "max_new_tokens=512" in text
    assert "runs-on: ubuntu-latest" in text
    assert "pause_space" in text
    assert "HF ZeroGPU" not in text.split("heldout-v3c:", 1)[-1]


def test_v3c_workflow_reuses_frozen_reference_instead_of_rerunning_base():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'run-id: "38423661192"' in text
    assert 'run-id: "38426519870"' in text
    assert "run_arm base" not in text
    assert "run_arm v3b" not in text
