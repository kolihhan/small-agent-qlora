"""Private experimental ZeroGPU Qwen3.5-9B 4K QLoRA compatibility pilot.

Never promote an adapter merely because a checkpoint exists.
Only source-publisher verified search/fetch state windows from original
Apache TRAIN source; no GAIA or source VALIDATION questions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
import traceback

import gradio as gr
import spaces
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from gaia_small_agent.training.qlora import (
    _render_turn_examples,
    _tokenize_completion_examples,
    load_verified_rows,
    train_qlora,
)

BASE_MODEL="Qwen/Qwen3.5-9B"
MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MAX_LENGTH=4096
MAX_STEPS=8
SAVE_STEPS=4
GRAD_ACC=1
DATA_SHA256="5420cfdb6111806a23b4bf6cb9cf04c77c75b5f0cf4af6c0b63727f312f40432"
PROTOCOL="nextsearch-balanced-true-source-4096-qlora-v0"
ROOT=Path("/tmp/nextsearch-real-states-9b-v0")
DATA=Path(__file__).parent/"nextsearch-balanced-source-states-4k.jsonl"
MODEL_PATH=snapshot_download(BASE_MODEL,revision=MODEL_REVISION,token=os.environ.get("HF_TOKEN"))
TOKENIZER=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)


def preflight() -> dict:
    from hashlib import sha256
    import torch
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    if sha256(DATA.read_bytes()).hexdigest()!=DATA_SHA256:
        raise ValueError("original GitHub state dataset source SHA drift")
    rows=load_verified_rows(DATA)
    if len(rows)!=48:
        raise ValueError(f"frozen exposure count mismatch: {len(rows)}")
    states={}
    parents={}
    for row in rows:
        state=row.get("capability")
        states[state]=states.get(state,0)+1
        parent=row.get("source_parent_sha256")
        parents[parent]=parents.get(parent,0)+1
        if (row.get("source_split")!="train"
            or row.get("original_teacher_action_unchanged") is not True
            or row.get("task_success_independently_replayed") is not False):
            raise ValueError("unverified source teaching decision")
    if states!={"evidence_missing_continue":24,"evidence_present_finalize":24}:
        raise ValueError("not equal missing evidence/final distribution")
    if len(parents)!=28 or max(parents.values())>2:
        raise ValueError("source parent breadth/weight drift")
    rendered=_render_turn_examples(TOKENIZER,rows)
    tokenized=_tokenize_completion_examples(
        TOKENIZER,rendered,max_length=MAX_LENGTH,require_full_prompt=True
    )
    if len(tokenized)!=48 or any(
        not any(v!=-100 for v in item["labels"]) for item in tokenized
    ):
        raise ValueError("loss mask or true context tokenization invalid")
    return {
        "source_sha256":DATA_SHA256,
        "source_parents":28,"original_source_next_actions":len(rendered),
        "states":states,
        "model_revision":MODEL_REVISION,
        "full_prompt_completion_retained":True,
        "max_length":MAX_LENGTH,"steps":MAX_STEPS,"save_steps":SAVE_STEPS,
        "gpu_available":torch.cuda.is_available(),
        "training_started":False,
        "source_label_only_not_independent_reward":True,
        "gaia_success_claimed":False,
    }


@spaces.GPU(duration=195)
def train():
    try:
        import torch
        check=preflight()
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable; refuse fake training")
        ROOT.mkdir(parents=True,exist_ok=True)
        adapter=ROOT/"adapter"
        start=time.monotonic()
        train_qlora(
            data_path=DATA,
            output_dir=adapter,
            model_name=MODEL_PATH,
            max_length=MAX_LENGTH,
            learning_rate=5e-5,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            max_steps=MAX_STEPS,
            save_steps=SAVE_STEPS,
            gradient_accumulation_steps=GRAD_ACC,
            lora_target_modules="all-linear",
            exposure_count=None,
            require_full_prompt=True,
        )
        required=[
            adapter/"adapter_model.safetensors",
            adapter/"adapter_config.json",
            adapter/"run_config.json"
        ]
        missing=[str(f) for f in required if not f.is_file()]
        if missing:
            raise RuntimeError("no verified adapter output: "+str(missing))
        config=json.loads((adapter/"run_config.json").read_text())
        if (
            config.get("training_examples")!=48
            or config.get("max_length")!=4096
            or config.get("require_full_prompt") is not True
            or config.get("max_steps")!=MAX_STEPS
        ):
            raise ValueError("adapter checkpoint training contract drift")
        output=shutil.make_archive(str(ROOT/"nextsearch9b-real-states-v0"),"zip",adapter)
        meta={
            **check,"ok":True,
            "step_target":MAX_STEPS,
            "seconds_elapsed":round(time.monotonic()-start,2),
            "adapter_bytes":(adapter/"adapter_model.safetensors").stat().st_size,
            "gpu":torch.cuda.get_device_name(0),
            "train_compute":True,
            "validation_performed":False,
            "promoted":False,
        }
        (ROOT/"training-result.json").write_text(json.dumps(meta,indent=2)+"\n")
        return output,meta
    except Exception as exc:
        return None,{
            "ok":False,"error":str(exc),"type":type(exc).__name__,
            "traceback":traceback.format_exc(),
            "adapter_promoted":False,
        }


def saved_checkpoint():
    """Recover checkpoint if timed training leaves one on a live Space worker."""
    out=ROOT/"adapter"
    files=list(out.glob("checkpoint-*/adapter_model.safetensors")) if out.exists() else []
    files+=list(out.glob("adapter_model.safetensors")) if out.exists() else []
    if not files:
        return None,{"ok":False,"reason":"no durable adapter file yet"}
    archive=shutil.make_archive(str(ROOT/"nextsearch9b-partial-state-checkpoint"),"zip",out)
    return archive,{"ok":True,"source_sha256":DATA_SHA256,
                    "checkpoint_file_count":len(files),"promoted":False}


with gr.Blocks() as demo:
    with gr.Row():
        checker=gr.Button("Preflight (no GPU)")
        pre_out=gr.JSON()
        checker.click(preflight,inputs=[],outputs=pre_out,api_name="preflight")
    with gr.Row():
        start=gr.Button("Train 8 real optimizer steps (free ZeroGPU)")
        train_file=gr.File()
        train_metadata=gr.JSON()
        start.click(train,inputs=[],outputs=[train_file,train_metadata],api_name="train")
    with gr.Row():
        recover=gr.Button("Recover checkpoint (no GPU)")
        checkpoint_file=gr.File()
        checkpoint_meta=gr.JSON()
        recover.click(saved_checkpoint,inputs=[],outputs=[checkpoint_file,checkpoint_meta],
                      api_name="checkpoint")


if __name__=="__main__":
    demo.launch(show_error=True)
