"""New v3E training is isolated and comparable with the successful frozen v3B."""
from pathlib import Path
import re


def test_v3e_training_workflow_only_changes_known_bad_action_labels():
    flow = Path(".github/workflows/hf-zero-gpu-train-v3e.yml").read_text(encoding="utf-8")
    assert re.search(r'EXPERIMENT_COMMIT: "[0-9a-f]{40}"', flow)
    assert 'TRAIN_STEPS: "64"' in flow
    assert 'EXPOSURE_COUNT: "64"' in flow
    assert 'GRAD_ACC: "1"' in flow
    assert 'MAX_LENGTH: "1024"' in flow
    assert 'EXPOSURE_SEED: "qlora-v3b-exposure"' in flow
    assert 'EXPOSURE_PROFILE: "v3e-cleaned"' in flow
    assert "_select_v3e_clean_exposure_examples" in flow
    assert "lora_target_modules='all-linear'" in flow
    assert "qwen35-9b-qlora-v3e-adapter.zip" in flow
    assert "hf-zero-gpu-train-v3b.yml" not in flow
    assert "kolihhan/small-agent-qlora-v2-train" not in flow
    assert "kolihhan/small-agent-qlora-v3e-train" in flow
    assert "api.create_repo(" in flow
    assert ".github/QLORA_V3E_TRIGGER" in flow
    assert "HF_TOKEN" in flow
    assert "qwen3.5" in flow.casefold()
