"""Static safety and reproducibility contract for the Colab GPU v3E notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re


NOTEBOOK = Path("notebooks/colab_qlora_v3e_clean_ablation.ipynb")
EXPECTED_SOURCE = "204dab90bd37692d59cdde344c3b9ff594d5f4e2"


def _source_cells():
    content = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert content["nbformat"] == 4
    assert content["metadata"]["accelerator"] == "GPU"
    assert len(content["cells"]) >= 8
    assert all(cell["cell_type"] in {"markdown", "code"} for cell in content["cells"])
    assert all(not cell.get("outputs") for cell in content["cells"])
    code = []
    for cell in content["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        ast.parse(source, filename=str(NOTEBOOK))
        code.append(source)
    return code


def test_colab_notebook_parses_and_cannot_claim_training_before_model_save():
    code = _source_cells()
    assert len(code) >= 7
    entire = "\n".join(code)
    assert f'SOURCE_SHA = "{EXPECTED_SOURCE}"' in entire
    assert 'checkout", "--detach", SOURCE_SHA' in entire
    assert "assert actual == SOURCE_SHA" in entire
    assert "assert torch.cuda.is_available()" in entire
    assert 'assert not ADAPTER.exists()' in entire
    assert 'profile="v3e-cleaned"' in entire or 'exposure_profile="v3e-cleaned"' in entire
    assert "deliberately_failed_first_call" in entire
    assert "sum(deliberately_failed_first_call(x) for x in old) == 4" in entire
    assert "sum(deliberately_failed_first_call(x) for x in clean) == 0" in entire
    assert "hashlib.file_digest" in entire
    assert "drive.mount" in entire
    assert "from google.colab import drive" in entire
    assert "v3b-diagnostic" in entire or "_select_diagnostic_exposure_examples" in entire
    assert "require_full_prompt=True" in entire
    assert "set_cuda_visible_devices" not in entire
    assert "model.push_to_hub" not in entire
    assert "secrets.HF_TOKEN" not in entire


def test_colab_pre_registered_training_ablation_matches_frozen_v3b():
    code = "\n".join(_source_cells())
    assert re.search(r'model_name=MODEL', code)
    assert 'MODEL = "Qwen/Qwen3.5-9B"' in code
    for fragment in (
        'max_length=1024', 'learning_rate=1e-4', 'lora_r=16',
        'lora_alpha=32', 'lora_dropout=0.05',
        'max_steps=64', 'save_steps=64',
        'gradient_accumulation_steps=1', 'lora_target_modules="all-linear"',
        'exposure_count=64', 'exposure_seed="qlora-v3b-exposure"',
        'exposure_profile="v3e-cleaned"', 'require_full_prompt=True',
        'seed="small-agent-policy-v3"',
    ):
        assert fragment in code, f"Missing or drifted setting: {fragment}"
    assert 'train_qlora(' in code
    assert 'gaia-protected' not in code
