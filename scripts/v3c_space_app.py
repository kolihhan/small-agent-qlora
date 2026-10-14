from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from gaia_small_agent.training.exposure_v3c import select_v3c_trajectories, summarize_v3c_selection
from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
from gaia_small_agent.training.qlora import (
    _render_turn_examples,
    _tokenize_completion_examples,
    load_verified_rows,
    train_qlora,
)

MODEL = "Qwen/Qwen3.5-9B"
ROOT = Path("/tmp/qlora-v3c")
MODEL_PATH = snapshot_download(repo_id=MODEL, token=os.environ.get("HF_TOKEN") or None)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "128"))
SELECTION_SEED = os.environ.get("SELECTION_SEED", "qlora-v3c-full-trajectory")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "1"))

EXPECTED_CAPABILITY_EXAMPLES = {
    "argument_grounding": 12,
    "duplicate_avoidance": 4,
    "evidence_to_final": 20,
    "failure_recovery": 36,
    "multi_step": 24,
    "no_tool_stop": 6,
    "strategy_switch": 18,
    "tool_selection": 8,
}
EXPECTED_ACTIONS = {"final": 54, "tool_call": 74}


def _build_selected_data() -> tuple[Path, dict]:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    paths = generate_oracle_policy_v3_dataset(ROOT / "oracle", seed="small-agent-policy-v3")
    source_rows = load_verified_rows(paths["train"])
    source_turns = _render_turn_examples(TOKENIZER, source_rows)
    selected_rows = select_v3c_trajectories(source_rows, seed=SELECTION_SEED)
    selected_turns = _render_turn_examples(TOKENIZER, selected_rows)
    report = summarize_v3c_selection(selected_rows)
    report.update({
        "selection_seed": SELECTION_SEED,
        "source_verified_rows": len(source_rows),
        "source_example_count": len(source_turns),
        "rendered_selected_example_count": len(selected_turns),
    })

    selected_path = ROOT / "train-v3c.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    (ROOT / "v3c-selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected_path, report


def preflight() -> dict:
    selected_path, report = _build_selected_data()
    rows = load_verified_rows(selected_path)
    turns = _render_turn_examples(TOKENIZER, rows)
    lengths = []
    for item in turns:
        prompt = len(TOKENIZER(item["prompt"], add_special_tokens=False)["input_ids"])
        completion = len(TOKENIZER(item["completion"], add_special_tokens=False)["input_ids"])
        lengths.append((prompt, completion, prompt + completion))
    tokenized = _tokenize_completion_examples(
        TOKENIZER,
        turns,
        max_length=MAX_LENGTH,
        require_full_prompt=True,
    )
    supervised = [sum(1 for token in row["labels"] if token != -100) for row in tokenized]
    report.update({
        "tokenized_examples": len(tokenized),
        "minimum_safe_context": max(total for _, _, total in lengths),
        "max_length": MAX_LENGTH,
        "min_supervised_tokens": min(supervised),
        "max_supervised_tokens": max(supervised),
        "all_have_supervised_tokens": all(value > 0 for value in supervised),
        "require_full_prompt": True,
        "training_steps": TRAIN_STEPS,
        "gradient_accumulation_steps": GRAD_ACC,
    })
    return report


@spaces.GPU(duration=195)
def train():
    try:
        if TRAIN_STEPS != 128:
            raise ValueError(f"TRAIN_STEPS must stay frozen at 128, got {TRAIN_STEPS}")
        if MAX_LENGTH != 1024:
            raise ValueError(f"MAX_LENGTH must stay frozen at 1024, got {MAX_LENGTH}")
        if GRAD_ACC != 1:
            raise ValueError(f"GRAD_ACC must stay frozen at 1, got {GRAD_ACC}")
        if SELECTION_SEED != "qlora-v3c-full-trajectory":
            raise ValueError(f"unexpected SELECTION_SEED: {SELECTION_SEED}")

        selected_path, selection = _build_selected_data()
        out = ROOT / "adapter"
        started = time.time()
        train_qlora(
            data_path=selected_path,
            output_dir=out,
            model_name=MODEL_PATH,
            max_length=MAX_LENGTH,
            learning_rate=1e-4,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            max_steps=TRAIN_STEPS,
            save_steps=TRAIN_STEPS,
            gradient_accumulation_steps=GRAD_ACC,
            lora_target_modules="all-linear",
            exposure_count=None,
            require_full_prompt=True,
        )
        elapsed = time.time() - started

        required = [
            out / "adapter_config.json",
            out / "adapter_model.safetensors",
            out / "run_config.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing adapter outputs: {missing}")

        cfg = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        if cfg.get("training_examples") != 128 or cfg.get("source_training_examples") != 128:
            raise RuntimeError(f"unexpected training counts: {cfg}")
        if cfg.get("max_length") != 1024 or cfg.get("require_full_prompt") is not True:
            raise RuntimeError(f"full-context contract was not preserved: {cfg}")
        if selection.get("source_example_count") != 3600:
            raise RuntimeError(f"unexpected oracle source exposure: {selection}")
        if selection.get("selected_task_count") != 54 or selection.get("selected_example_count") != 128:
            raise RuntimeError(f"unexpected v3C selection size: {selection}")
        if selection.get("action_type_counts") != EXPECTED_ACTIONS:
            raise RuntimeError(f"unexpected v3C action distribution: {selection}")
        if selection.get("capability_example_counts") != EXPECTED_CAPABILITY_EXAMPLES:
            raise RuntimeError(f"unexpected v3C capability distribution: {selection}")

        shutil.copyfile(ROOT / "v3c-selection.json", out / "v3c-selection.json")
        archive = shutil.make_archive(str(ROOT / "qwen35-9b-qlora-v3c-adapter"), "zip", out)
        meta = {
            "ok": True,
            "model": MODEL,
            "experiment_commit": os.environ.get("EXPERIMENT_COMMIT"),
            "max_steps": TRAIN_STEPS,
            "max_length": MAX_LENGTH,
            "gradient_accumulation_steps": GRAD_ACC,
            "learning_rate": 1e-4,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "all-linear",
            "elapsed_seconds": elapsed,
            "seconds_per_optimizer_step": elapsed / TRAIN_STEPS,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else 0,
            "adapter_bytes": (out / "adapter_model.safetensors").stat().st_size,
            "training_examples": cfg.get("training_examples"),
            "source_training_examples": cfg.get("source_training_examples"),
            "oracle_source_example_count": selection.get("source_example_count"),
            "selected_task_count": selection.get("selected_task_count"),
            "selected_example_count": selection.get("selected_example_count"),
            "selection_sha256": selection.get("selection_sha256"),
            "action_type_counts": selection.get("action_type_counts"),
            "capability_example_counts": selection.get("capability_example_counts"),
            "truncation_policy": cfg.get("truncation_policy"),
            "require_full_prompt": cfg.get("require_full_prompt"),
        }
        return archive, meta
    except Exception as exc:
        return None, {
            "ok": False,
            "type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


with gr.Blocks() as demo:
    pre = gr.Button("preflight")
    pre_out = gr.JSON()
    pre.click(preflight, inputs=[], outputs=pre_out, api_name="preflight")
    run = gr.Button("train")
    out_file, out_json = gr.File(), gr.JSON()
    run.click(train, inputs=[], outputs=[out_file, out_json], api_name="train")


if __name__ == "__main__":
    demo.launch(show_error=True)
