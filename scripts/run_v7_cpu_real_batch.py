"""Real v7 Qwen3.5-9B QLoRA continuation on GitHub Actions CPU, one batch.

Unlike the previous synthetic v3D CPU continuation, this consumes the audited
522-action v7 source with full observed tool traces, source-balanced and parent
disjoint. Trains LoRA layers 30+31 only to fit commodity CPU RAM; this is an
*experimental capacity tradeoff*, not equivalent to all-linear QLoRA SFT.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-index",type=int,required=True)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--source-adapter",type=Path,required=True)
    p.add_argument("--source-state",type=Path)
    p.add_argument("--source-optimizer",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--learning-rate",type=float,default=5e-5)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("do not overwrite old training checkpoint")
    if not a.source_adapter.is_dir():
        raise FileNotFoundError(a.source_adapter)
    if (a.source_state is None) != (a.source_optimizer is None):
        raise ValueError("resume requires both state and optimizer")
    if (a.batch_index==0) != (a.source_state is None):
        raise ValueError("batch 0 starts from verified v3B; later batches must resume optimizer")

    from scripts.qlora_v7_hf_chunk import build_free_gpu_chunk
    from gaia_small_agent.training.qlora import (
        _render_turn_examples,_tokenize_completion_examples,
    )
    audit=json.loads(a.audit.read_text())
    if (audit.get("schema")!="qlora-v7-whole-task-source-balanced-20261113-v1"
        or audit.get("train_actions")!=522 or audit.get("train_tasks")!=218
        or audit.get("holdout_tasks")!=11 or audit.get("train_real_action_fraction")!=0.333333):
        raise ValueError("v7 data frozen source contract is not satisfied")
    if _checksum(a.input)!=audit["train_sha256"]:
        raise ValueError("v7 source sha mismatch")
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    if len(rows)!=218:
        raise ValueError("218 unique parent actions required")
    current=build_free_gpu_chunk(rows,a.batch_index)
    prior_ids=[]
    prior_steps=0
    if a.source_state:
        state=json.loads(a.source_state.read_text())
        if state.get("schema")!="qlora-v7-cpu-resumable-loRa-v1":
            raise ValueError("wrong prior continuation protocol")
        if state.get("batch_index")!=a.batch_index-1 or state.get("train_sha256")!=audit["train_sha256"]:
            raise ValueError("wrong predecessor state")
        prior_ids=list(state["completed_parent_ids"])
        expected=[r["task_id"] for i in range(a.batch_index)
                  for r in build_free_gpu_chunk(rows,i)]
        if prior_ids!=expected:
            raise ValueError("source checkpoint parent exposure overlap or gaps")
        prior_steps=int(state["completed_optimizer_steps"])
    current_ids=[r["task_id"] for r in current]
    if set(prior_ids)&set(current_ids):
        raise ValueError("repeated parent data across batches")

    import torch
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if torch.cuda.is_available():
        raise RuntimeError("this experiment is CPU only; do not silently use GPU")
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model_name="Qwen/Qwen3.5-9B"
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    examples=_render_turn_examples(tokenizer,current)
    tokenized=_tokenize_completion_examples(
        tokenizer,examples,max_length=1536,require_full_prompt=True)
    expected_count=sum(len(r["training_action_indices"]) for r in current)
    if len(tokenized)!=expected_count or not tokenized:
        raise ValueError("incomplete native next-assistant-action labels")
    metadata=[str(x["metadata"]["example_id"]) for x in examples]
    if len(metadata)!=len(set(metadata)):
        raise ValueError("duplicate assistant action supervision")
    quant=BitsAndBytesConfig(
        load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base=AutoModelForCausalLM.from_pretrained(
        model_name,quantization_config=quant,device_map="auto",
        trust_remote_code=True,
    )
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(base,str(a.source_adapter),is_trainable=True)
    trainable=[]
    for name,param in model.named_parameters():
        allowed=("lora_" in name and
                 (".layers.30." in name or ".layers.31." in name))
        param.requires_grad=allowed
        if allowed:
            trainable.append((name,param))
    if not trainable:
        raise RuntimeError("verified v3B PEFT has no layer30/31 LoRA tensors")
    optimizer=torch.optim.AdamW([q for _,q in trainable],lr=a.learning_rate)
    if a.source_optimizer:
        optimizer.load_state_dict(torch.load(
            a.source_optimizer,map_location="cpu",weights_only=True))
    a.output.mkdir(parents=True,exist_ok=False)
    model.train()
    history=[]
    started=time.monotonic()
    for i,row in enumerate(tokenized,1):
        if not any(v!=-100 for v in row["labels"]):
            raise ValueError("a supervised sample has no assistant loss")
        batch={k:torch.tensor([v],dtype=torch.long) for k,v in row.items()}
        t0=time.perf_counter()
        loss=model(**batch).loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("nonfinite real-action QLoRA loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        item={
            "batch":a.batch_index,
            "global_step":prior_steps+i,
            "action_id":metadata[i-1],
            "loss":float(loss.detach().cpu()),
            "seconds":round(time.perf_counter()-t0,2),
            "context_tokens":len(row["input_ids"]),
        }
        history.append(item)
        print("V7_CPU_REAL_QLO_STEP",json.dumps(item,sort_keys=True),flush=True)
    adapter_path=a.output/"checkpoint"/"adapter"
    adapter_path.mkdir(parents=True)
    model.save_pretrained(adapter_path)
    optimizer_path=a.output/"checkpoint"/"optimizer.pt"
    torch.save(optimizer.state_dict(),optimizer_path)
    weights=adapter_path/"adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size<50000000:
        raise RuntimeError("no valid full adapter output from CPU training")
    state={
        "schema":"qlora-v7-cpu-resumable-loRa-v1",
        "train_sha256":audit["train_sha256"],
        "source":"real observed 2Wiki+Hotpot plus synthetic eight-capability policy",
        "batch_index":a.batch_index,
        "total_batches":22,
        "batch_parent_count":len(current_ids),
        "batch_action_count":len(history),
        "completed_optimizer_steps":prior_steps+len(history),
        "completed_parent_ids":prior_ids+current_ids,
        "batch_parent_ids":current_ids,
        "trainable_scope":"layers 30 and 31 LoRA (CPU RAM bound)",
        "trainable_parameter_count":sum(p.numel() for _,p in trainable),
        "learning_rate":a.learning_rate,
        "max_length":1536,
        "train_only":True,
        "heldout_train_count":0,
        "gaia_train_count":0,
        "model_trained":True,
        "adapter_sha256":_checksum(weights),
        "adapter_bytes":weights.stat().st_size,
        "optimizer_sha256":_checksum(optimizer_path),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "loss_first":history[0]["loss"],
        "loss_last":history[-1]["loss"],
        "gaia_evaluated":False,
        "generalization_verified":False,
    }
    (a.output/"checkpoint"/"state.json").write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
    (a.output/"timings.json").write_text(json.dumps(history,indent=2,sort_keys=True)+"\n")
    print("QLORA_V7_CPU_PHYSICAL_ADAPTER_SAVED",json.dumps(state,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
