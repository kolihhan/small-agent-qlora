"""Kaggle T4x2 execution of *actual* v7 Qwen3.5-9B QLoRA.

Input: immutable, private Kaggle dataset containing qlora-v7-train.jsonl,
qlora-v7-heldout.jsonl and their SHA-256 audit manifest.
No GAIA labels/answers are imported. No SQL training is performed.

The CI bridge substitutes the exact source SHA and dataset slug once before
uploading this file as a PRIVATE Kaggle kernel. The script never uploads model
weights to an external provider or silently replays a failed optimizer run.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE_SHA = "__QLORA_SOURCE_SHA__"
DATASET_SLUG = "__QLORA_DATASET_SLUG__"
MODEL = "Qwen/Qwen3.5-9B"
WORK = Path("/kaggle/working")
OUT = WORK / "qlora-v7-observed-tool-sft"
STEPS = 65


def cmd(*args):
    print("EXEC", args[0], args[1] if len(args) > 1 else "", flush=True)
    subprocess.run(list(args), check=True)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("KAGGLE_GPU_NOT_ALLOCATED; refusing CPU substitute")
    vram = torch.cuda.get_device_properties(0).total_memory/(1024**3)
    if vram < 14.5:
        raise RuntimeError("T4 or better GPU with >=14.5 GiB required")
    if SOURCE_SHA.startswith("__") or DATASET_SLUG.startswith("__"):
        raise RuntimeError("trusted CI failed to pin immutable source and input")
    root = Path("/kaggle/input") / DATASET_SLUG
    train = root / "qlora-v7-train.jsonl"
    heldout = root / "qlora-v7-heldout.jsonl"
    audit = root / "qlora-v7-audit.json"
    if not all(p.is_file() for p in (train, heldout, audit)):
        raise RuntimeError("private Kaggle source dataset missing")
    report = json.loads(audit.read_text(encoding="utf-8"))
    if (report.get("schema") != "qlora-v7-whole-task-source-balanced-20261113-v1"
        or report.get("train_actions") != 522
        or report.get("train_tasks") != 218
        or report.get("holdout_tasks") != 11
        or abs(report.get("train_real_action_fraction",0)-1/3)>1e-5):
        raise ValueError("v7 source contract drift; refuse train")
    for path,key in ((train,"train_sha256"),(heldout,"holdout_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
            raise ValueError(f"{path.name} does not match validated CPU data")
    print("V7_IMMUTABLE_DATA_VERIFIED",json.dumps({
        "gpu":torch.cuda.get_device_name(0),"vram_gib":round(vram,2),
        "num_cuda_devices":torch.cuda.device_count(),
        "train_actions":report["train_actions"],
        "real_action_fraction":report["train_real_action_fraction"],
        "heldout_parents":report["holdout_tasks"],
    },sort_keys=True),flush=True)

    repo=WORK/"small-agent-qlora-v7-frozen"
    if repo.exists() or OUT.exists():
        raise FileExistsError("refuse to overwrite prior training checkpoint")
    cmd("git","clone","--depth","1","--branch",
        "experiment/qlora-v7-kaggle-gpu-20261113",
        "https://github.com/kolihhan/small-agent-qlora.git",str(repo))
    actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual!=SOURCE_SHA:
        raise ValueError("GitHub source SHA changed during Kaggle run")
    cmd(sys.executable,"-m","pip","install","-q","-e",str(repo)+"[train,dev,files]","jinja2>=3.1")
    sys.path.insert(0,str(repo))
    sys.path.insert(0,str(repo/"src"))
    from gaia_small_agent.training.qlora import train_qlora

    t0=time.time()
    # 522 unique labels. 65 optimizer steps x accumulation 8 = 520
    # supervised actions, about one source epoch; do not repeat tiny real data.
    train_qlora(
        data_path=train,output_dir=OUT,model_name=MODEL,
        max_length=1536,epochs=1.0,learning_rate=5e-5,
        lora_r=16,lora_alpha=32,lora_dropout=0.05,
        max_steps=STEPS,save_steps=25,gradient_accumulation_steps=8,
        lora_target_modules="all-linear",exposure_count=None,
        require_full_prompt=True,preserve_large_half_parameters=(vram<20),
    )
    safetensors=OUT/"adapter_model.safetensors"
    if not safetensors.is_file() or safetensors.stat().st_size<1000000:
        raise RuntimeError("no physically verified PEFT adapter")
    cfg=json.loads((OUT/"run_config.json").read_text())
    if (cfg.get("training_examples")!=522 or cfg.get("max_steps")!=STEPS
        or cfg.get("require_full_prompt") is not True
        or cfg.get("lora_target_modules")!="all-linear"):
        raise ValueError("trainer violated declared v7 full-LoRA experiment contract")
    stamp={
        "schema":"qlora-v7-kaggle-trained-adapter-v1",
        "model":MODEL,"source_sha":SOURCE_SHA,
        "train_sha256":report["train_sha256"],
        "holdout_sha256":report["holdout_sha256"],
        "real_fraction":report["train_real_action_fraction"],
        "train_actions":522,"train_parents":218,"heldout_parents":11,
        "optimizer_steps":STEPS,
        "weights_sha256":hashlib.sha256(safetensors.read_bytes()).hexdigest(),
        "weights_bytes":safetensors.stat().st_size,
        "training_seconds":round(time.time()-t0,2),
        "gaia_evaluated":False,
        "notes":"Kaggle GPU training only. Independent GAIA held out; no generalization claimed.",
    }
    (OUT/"verified-kaggle-adapter-report.json").write_text(json.dumps(stamp,indent=2,sort_keys=True)+"\n")
    print("V7_KAGGLE_9B_ADAPTER_VERIFIED",json.dumps(stamp,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
