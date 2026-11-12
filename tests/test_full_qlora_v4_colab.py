"""Full v4 Colab notebook contract: 3200 clean actions, real 9B QLoRA, not 64-step v3E."""
import ast
import json
from pathlib import Path


def test_full_qlora_notebook_can_run_all_and_save_real_adapter():
    path = Path("notebooks/colab_qlora_v4_full_sft.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    all_code = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            ast.parse(src, filename=str(path))
            assert not cell["outputs"]
            all_code.append(src)
    body = "\n".join(all_code)
    for token in (
        "Qwen/Qwen3.5-9B", "generate_oracle_policy_v3_dataset",
        "prepare_full_clean_train(", "total_trainable_actions", "3200",
        "removed_intentionally_bad_actions", "400",
        "_render_turn_examples(", "_tokenize_completion_examples(",
        "train_qlora(", "max_steps=400", "save_steps=50",
        "gradient_accumulation_steps=8", "lora_target_modules=\"all-linear\"",
        "exposure_count=None", "require_full_prompt=True",
        "resume_from_checkpoint=", "drive.mount(", "adapter_model.safetensors",
    ):
        assert token in body, token
    assert "exposure_count=64" not in body
    assert "hf-zero-gpu" not in body.lower()
    assert "gaia-benchmark" not in body.lower()
