"""2-step memory feasibility ONLY: real-source full-context Qwen3.5-9B on T4.

Do not launch while the independent full-v4 Shadow25 benchmark is queued.
Source data is the *already verified* 39-complete-task subset. No GAIA
validation data is used for training (only protected question hashes).
Output is a small proof-of-compute report, not a scored trained agent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

MODEL="Qwen/Qwen3.5-9B"
PINNED_SOURCE_SHA="2e51b2a6b7a4be6fc723a0dfbf35f5629b8ff350"
DATA=Path("/content/real-multisource-2048-complete-tasks.jsonl")
PROTECTION=Path("/content/gaia-protected-hashes.json")
EXPECTED_SHA="79c22f0ef7bb6015eeb3997d11f302caf6fc77667d1494813a315ae41ceab676"
OUT=Path("/content/real-2048-smoke-adapter")
REPORT=Path("/content/real-2048-smoke-report.json")


def _exec(*cmd:str)->None:
    subprocess.run(list(cmd),check=True)


def main()->None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    capacity=torch.cuda.get_device_properties(0).total_memory/(1024**3)
    if capacity<14.5:
        raise RuntimeError("T4 or larger accelerator required")
    if not DATA.is_file() or not PROTECTION.is_file():
        raise FileNotFoundError("missing protected, curated real task inputs")
    digest=hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA:
        raise ValueError("curated 39-task dataset identity changed")
    rows=[json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows)!=39 or len({str(r["task_id"]) for r in rows})!=39:
        raise ValueError("require 39 independent complete tasks")
    if sum(len(r["training_action_indices"]) for r in rows)!=154:
        raise ValueError("require all 154 native actions")
    if {r["source"] for r in rows}!={"hotpotqa-distractor-train","2wikimultihopqa-train"}:
        raise ValueError("unexpected source / possible protected evaluation leakage")
    root=Path("/content/small-agent-qlora")
    if root.exists() or OUT.exists() or REPORT.exists():
        raise FileExistsError("refuse stale source/model artifacts")
    _exec("git","clone","--branch","fix/qlora-v4-t4-4bit-memory-20261112",
          "--depth","100","https://github.com/kolihhan/small-agent-qlora.git",str(root))
    _exec("git","-C",str(root),"checkout","--detach",PINNED_SOURCE_SHA)
    if subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()!=PINNED_SOURCE_SHA:
        raise RuntimeError("unpinned runtime")
    _exec(sys.executable,"-m","pip","install","-q","-e",str(root)+"[train,dev,files]","jinja2>=3.1")
    sys.path[:0]=[str(root),str(root/"src")]
    from gaia_small_agent.training.qlora import (
        load_verified_rows,_render_turn_examples,
        _tokenize_completion_examples,train_qlora)
    from transformers import AutoTokenizer
    protected=load_verified_rows(DATA,protected_questions_path=PROTECTION)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    rendered=_render_turn_examples(tokenizer,protected)
    if len(rendered)!=154: raise ValueError("turn renderer dropped a native assistant action")
    tok=_tokenize_completion_examples(
        tokenizer,rendered,max_length=2048,require_full_prompt=True)
    if len(tok)!=154: raise ValueError("tokenizer dropped assistant action")
    print("REAL_2048_GPU_PREFLIGHT",json.dumps({
        "accelerator":torch.cuda.get_device_name(0),
        "vram_gib":round(capacity,2),"task_count":39,
        "train_actions":len(tok),"max_length":max(len(x["input_ids"]) for x in tok),
        "data_sha256":digest,"full_context_preserved":True
    },sort_keys=True),flush=True)
    start=time.monotonic()
    train_qlora(
        data_path=DATA,output_dir=OUT,model_name=MODEL,max_length=2048,
        epochs=1.0,learning_rate=1e-4,
        lora_r=16,lora_alpha=32,lora_dropout=0.05,
        protected_questions_path=PROTECTION,
        max_steps=2,save_steps=2,gradient_accumulation_steps=1,
        lora_target_modules="all-linear",exposure_count=None,
        require_full_prompt=True,preserve_large_half_parameters=(capacity<20)
    )
    weight=OUT/"adapter_model.safetensors"
    cfg=json.loads((OUT/"run_config.json").read_text(encoding="utf-8"))
    if (not weight.is_file() or weight.stat().st_size<=1_000_000
        or cfg["training_examples"]!=154
        or cfg["max_steps"]!=2 or cfg["max_length"]!=2048
        or cfg["require_full_prompt"] is not True):
        raise RuntimeError("real 2048 two-step adapter not independently verified")
    report={
        "schema":"real-2048-qwen35-9b-t4-2step-feasibility-v1",
        "gpu":torch.cuda.get_device_name(0),
        "vram_gib":round(capacity,2),
        "data_sha256":digest,"dataset_tasks":39,"dataset_actions":154,
        "max_length":2048,"full_prompt":True,
        "optimizer_steps":2,"adapter_bytes":weight.stat().st_size,
        "adapter_sha256":hashlib.sha256(weight.read_bytes()).hexdigest(),
        "seconds":round(time.monotonic()-start,1),
        "gaia_score":"UNMEASURED",
        "training_goal":"memory-only smoke; NOT performance",
        "runtime_sha":PINNED_SOURCE_SHA,
    }
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("REAL_2048_GPU_SMOKE_VERIFIED",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
