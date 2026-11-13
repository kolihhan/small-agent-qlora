"""Hugging Face private ZeroGPU adapter continuation (one non-repeated chunk).

No Kaggle, no SQL training, no GAIA labels, no paid HF Jobs.
This uses the frozen, GH-verified v3B PEFT adapter as the starting point.
Only the 24-ish unseen action decisions from one of 22 whole-parent chunks
are supervised, using full original observations and next-action loss.
The returned checkpoint contains physically saved LoRA weights and optimizer
state so continuation can preserve learning across free ZeroGPU windows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import zipfile

import gradio as gr
import spaces
from transformers import AutoTokenizer

from gaia_small_agent.training.qlora import _render_turn_examples, _tokenize_completion_examples
from qlora_v7_hf_chunk import build_free_gpu_chunk

ROOT=Path(__file__).resolve().parent
MODEL="Qwen/Qwen3.5-9B"
MAX_LENGTH=1536
BATCH_INDEX=0
BASE_ADAPTER=ROOT/"v3b-adapter"
DATA=ROOT/"training-data"/"qlora-v7-train.jsonl"
AUDIT=ROOT/"training-data"/"qlora-v7-audit.json"
OUT=Path("/tmp/qlora-v7-hf-zero-gpu-batch-0")
RUN_STAMP="qlora-v7-hf-zero-gpu-real-parent-batch-0"
TOKENIZER=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)


def get_training_examples():
    if not DATA.is_file() or not AUDIT.is_file():
        raise FileNotFoundError("audited v7 train data missing")
    audit=json.loads(AUDIT.read_text())
    if (audit.get("train_actions")!=522 or audit.get("train_tasks")!=218
        or audit.get("holdout_tasks")!=11
        or audit.get("train_real_action_fraction")!=0.333333):
        raise ValueError("frozen source parent counts changed")
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=audit.get("train_sha256"):
        raise ValueError("source JSONL SHA-256 does not match observed 522-action audit")
    rows=[json.loads(x) for x in DATA.read_text().splitlines() if x.strip()]
    chosen=build_free_gpu_chunk(rows,BATCH_INDEX)
    if len(chosen)!=10 or len({r["task_id"] for r in chosen})!=10:
        raise ValueError("need ten unique parent tasks in first batch")
    if len([r for r in chosen if r["source"]!="synthetic-small-agent-policy-v3/train"])!=2:
        raise ValueError("need two distinct observed source tasks")
    examples=_render_turn_examples(TOKENIZER,chosen)
    tokens=_tokenize_completion_examples(
        TOKENIZER,examples,max_length=MAX_LENGTH,require_full_prompt=True)
    if len(tokens) not in (22,23,24):
        raise ValueError("unexpected first-batch action count")
    if not all(any(x!=-100 for x in row["labels"]) for row in tokens):
        raise ValueError("missing completion-only loss targets")
    return chosen,examples,tokens,audit


def preflight():
    chosen,examples,tokens,audit=get_training_examples()
    from collections import Counter
    caps=Counter(r.get("capability") or "observed-multihop" for r in chosen)
    by_source=Counter(r["source"] for r in chosen)
    return {
       "ok":True,
       "base":"previous verified Qwen3.5-9B v3B adapter",
       "new_supervised_native_actions":len(tokens),
       "whole_parent_tasks":len(chosen),
       "source_counts":dict(by_source),
       "capability_counts":dict(caps),
       "max_token_context":max(len(r["input_ids"]) for r in tokens),
       "max_length":MAX_LENGTH,
       "batch_index":BATCH_INDEX,
       "planned_total_batches":22,
       "gaia_train_count":0,
       "heldout_train_count":0,
       "source_sha256":audit["train_sha256"],
       "current_training_ran":False,
    }


@spaces.GPU(duration=240)
def train_one_batch():
    try:
        import torch
        from peft import PeftModel,prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM,BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("ZeroGPU failed to allocate CUDA; cannot call this a trained adapter")
        name=torch.cuda.get_device_name(0)
        vram=torch.cuda.get_device_properties(0).total_memory/(1024**3)
        if vram<20:
            raise RuntimeError("ZeroGPU under 20GB, cannot safely load full 9B PEFT continuation")
        if OUT.exists():
            raise FileExistsError("refuse overwrite or replay first batch of QLoRA SFT")
        if not (BASE_ADAPTER/"adapter_model.safetensors").is_file():
            raise FileNotFoundError("frozen verified v3B source weights missing")
        chosen,examples,tokens,audit=get_training_examples()
        OUT.mkdir(parents=True,exist_ok=False)
        start=time.time()
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        base=AutoModelForCausalLM.from_pretrained(
            MODEL,quantization_config=quant,device_map="auto",dtype=dtype,
            trust_remote_code=True)
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
        model=PeftModel.from_pretrained(base,str(BASE_ADAPTER),is_trainable=True)
        model.config.use_cache=False
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not named or any("lora_" not in n for n,p in named):
            raise ValueError("unexpected trainable parameter scope, want LoRA only")
        optimizer=torch.optim.AdamW([p for n,p in named],lr=5e-5)
        model.train()
        losses=[]
        timings=[]
        for index,item in enumerate(tokens,1):
            inputs={k:torch.tensor([v],dtype=torch.long,device="cuda:0")
                    for k,v in item.items()}
            before=time.monotonic()
            outputs=model(**inputs)
            loss=outputs.loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite QLoRA loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _,p in named],max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            timings.append(round(time.monotonic()-before,3))
            print("QLORA_V7_HF_REAL_OPTIMIZER_STEP",json.dumps({
                "step":index,"loss":round(losses[-1],6),
                "seconds":timings[-1],
                "tokens":len(item["input_ids"])
            }),flush=True)
        adapter_dir=OUT/"adapter"
        model.save_pretrained(adapter_dir)
        TOKENIZER.save_pretrained(adapter_dir)
        weights=adapter_dir/"adapter_model.safetensors"
        if not weights.is_file() or weights.stat().st_size<1000000:
            raise RuntimeError("physical trained adapter not saved")
        # Optimizer continuity: save exactly the same optimizer object for future
        # GH Actions batch resumption; do not just restart momentum every batch.
        torch.save(optimizer.state_dict(),OUT/"optimizer.pt")
        report={
            "schema":"qlora-v7-zerogpu-batch0-physical-adapter-v1",
            "base":"verified-v3b-peft-adapter",
            "gpu":name,"vram_gib":round(vram,2),
            "trainable_scope":"all existing v3B LoRA tensors",
            "trainable_tensors":len(named),
            "optimizer_steps":len(losses),
            "trained_unique_parent_tasks":len(chosen),
            "train_action_ids":[x["metadata"]["example_id"] for x in examples],
            "train_parent_ids":[x["task_id"] for x in chosen],
            "train_parent_repetitions":0,
            "source_sha256":audit["train_sha256"],
            "heldout_parents_used":0,
            "max_length":MAX_LENGTH,
            "loss_first":losses[0],"loss_last":losses[-1],
            "loss_mean":sum(losses)/len(losses),
            "step_seconds":timings,
            "elapsed_seconds":round(time.time()-start,2),
            "adapter_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
            "adapter_bytes":weights.stat().st_size,
            "gaia_evaluated":False,
            "generalization":"unverified; held-out and GAIA required before promotion",
        }
        (OUT/"training-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        archive=ROOT/f"{RUN_STAMP}.zip"
        if archive.exists():
            raise FileExistsError("refuse overwrite previously completed adapter ZIP")
        with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_STORED) as z:
            for path in sorted(OUT.rglob("*")):
                if path.is_file():
                    z.write(path,arcname=str(path.relative_to(OUT)))
        print("QLORA_V7_HF_ZERO_GPU_ADAPTER_SAVED",json.dumps({
           "optimizer_steps":len(losses),
           "adapter_sha256":report["adapter_sha256"],
           "archive_bytes":archive.stat().st_size
        }),flush=True)
        return str(archive),{"ok":True,"report":report}
    except Exception as e:
        print("QLORA_V7_HF_TRAINING_ERROR",str(e),flush=True)
        return None,{"ok":False,"error_type":type(e).__name__,
                     "error":str(e),"traceback":traceback.format_exc()[-6000:]}


with gr.Blocks() as demo:
    gr.Markdown("Private QLoRA 9B real-tool SFT: 1/22 source-disjoint free ZeroGPU batch")
    check=gr.Button("Verify training source (CPU)")
    check_output=gr.JSON()
    check.click(preflight,inputs=[],outputs=check_output,api_name="preflight")
    go=gr.Button("Train one source-disjoint full-parent batch on free ZeroGPU")
    result_file=gr.File()
    result_summary=gr.JSON()
    go.click(train_one_batch,inputs=[],outputs=[result_file,result_summary],api_name="train")

if __name__=="__main__":
    demo.launch(show_error=True)
