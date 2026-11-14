"""Train one Qwen3.5-9B v8 4-bit all-linear LoRA candidate on Colab T4 if allocated.

This is a NEW from-base QLoRA adapter, not continued v7 weights. v7 training
examples are replayed once along with verified FinQA and annotation-derived
MuSiQue data; heldout and GAIA are never used as supervision or hyperparameter
feedback. Called by a SHA-attested GitHub Actions Colab runner.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SOURCE_SHA="__SOURCE_SHA__"
BRANCH="experiment/qlora-multihop-oracle-supervision-20261115"
TRAIN=Path("/content/qlora-v8-train.jsonl")
AUDIT=Path("/content/qlora-v8-audit.json")
OUT=Path("/content/qlora-v8-adapter")
ARCHIVE=Path("/content/qlora-v8-adapter.zip")
MODEL="Qwen/Qwen3.5-9B"
MAX_CONTEXT=3200
MAX_STEPS=100


def checked(*cmd):
    print("QLORA_V8_STEP",cmd[0],cmd[1] if len(cmd)>1 else "",flush=True)
    subprocess.run(list(cmd),check=True)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU; do not mistake CPU artifact preflight for QLoRA")
    vram=torch.cuda.get_device_properties(0).total_memory/2**30
    if vram < 14:
        raise RuntimeError("V8 requires >=14 GiB VRAM for full untruncated 3200-token context on T4; no CPU fallback")
    if SOURCE_SHA=="__SOURCE_SHA__" or len(SOURCE_SHA)!=40:
        raise RuntimeError("GitHub SHA not attested")
    report=json.loads(AUDIT.read_text(encoding="utf-8"))
    if (report.get("protocol")!="qlora-v8-source-balanced-full-parent-provisional-v1"
        or not report.get("training_ready_no_gpu_yet")
        or report.get("train_parents")!=304 or report.get("train_actions")!=798
        or report.get("validation_parents")!=35 or report.get("validation_actions")!=176
        or report.get("annotation_oracle_training_actions")!=192
        or not (0.37<=report["train_final_action_fraction"]<=0.40)
        or report.get("token_context_max")!=4096
        or report.get("no_gaia_training") is not True
        or report.get("gaia_evaluated") is not False):
        raise ValueError("v8 source contract changed after candidate selection")
    if sha256(TRAIN.read_bytes()).hexdigest()!=report["train_sha256"]:
        raise ValueError("v8 train file SHA mismatch")
    print("V8_GPU_ATTEST",json.dumps({
        "source_sha":SOURCE_SHA,"train_actions":798,"train_parents":304,
        "gpu":torch.cuda.get_device_name(0),"vram_gib":round(vram,2),
        "context":MAX_CONTEXT,"steps":MAX_STEPS,
        "pretrain_eval":"NONE",
    }),flush=True)
    repo=Path("/content/qlora-v8-source")
    if any(p.exists() for p in [repo,OUT,ARCHIVE]):
        raise FileExistsError("refuse reuse of prior GPU workspace")
    checked("git","clone","--branch",BRANCH,"--depth","30",
            "https://github.com/kolihhan/small-agent-qlora.git",str(repo))
    checked("git","-C",str(repo),"checkout","--detach",SOURCE_SHA)
    actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual!=SOURCE_SHA:
        raise RuntimeError("source commit checkout mismatch")
    checked(sys.executable,"-m","pip","install","-q","-e",str(repo)+"[train,files]",
            "jinja2>=3.1","sentencepiece>=0.2")
    sys.path.insert(0,str(repo/"src"))
    from gaia_small_agent.training.qlora import train_qlora
    import inspect
    required=("data_path","output_dir","model_name","max_length",
              "epochs","learning_rate","lora_r","lora_alpha",
              "lora_dropout","max_steps","save_steps","gradient_accumulation_steps",
              "lora_target_modules","exposure_count","require_full_prompt")
    missing=sorted(set(required)-set(inspect.signature(train_qlora).parameters))
    if missing:
        raise RuntimeError("QLORA_V8_TRAIN_SIGNATURE_DRIFT:"+",".join(missing))
    print("QLORA_V8_TRAIN_SIGNATURE_VERIFIED",len(required),flush=True)

    start=time.monotonic()
    train_qlora(
        data_path=TRAIN,output_dir=OUT,model_name=MODEL,
        max_length=MAX_CONTEXT,epochs=1.0,learning_rate=2e-5,
        lora_r=16,lora_alpha=32,lora_dropout=0.05,
        max_steps=MAX_STEPS,save_steps=50,gradient_accumulation_steps=8,
        lora_target_modules="all-linear",exposure_count=None,
        require_full_prompt=True,
    )
    weights=OUT/"adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size<=1_000_000:
        raise RuntimeError("physical Qwen9B v8 LoRA checkpoint absent")
    cfg=json.loads((OUT/"run_config.json").read_text())
    if cfg["training_examples"]!=798 or cfg["max_steps"]!=MAX_STEPS:
        raise ValueError("SFT training exposure/step mismatch")
    digest=sha256(weights.read_bytes()).hexdigest()
    final={
        "protocol":"qlora-v8-from-base-multisource-nf4-sft-v1",
        "source_sha":SOURCE_SHA,"model":MODEL,
        "checkpoint_sha256":digest,"checkpoint_bytes":weights.stat().st_size,
        "train_sha256":report["train_sha256"],
        "train_actions":798,"train_parents":304,
        "holdout_parents_unseen":35,
        "max_context":MAX_CONTEXT,"optimizer_steps":MAX_STEPS,
        "learning_rate":2e-5,"adapter_initialized":"from-base-not-v7",
        "duration_s":round(time.monotonic()-start),
        "gaia_evaluated":False,"external_generalization":"NOT_EVALUATED",
    }
    (OUT/"v8-verified-report.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    with zipfile.ZipFile(ARCHIVE,"x",compression=zipfile.ZIP_STORED) as z:
        for name in ("adapter_model.safetensors","adapter_config.json","run_config.json",
                     "v8-verified-report.json"):
            z.write(OUT/name,arcname=name)
    print("V8_PHYSICAL_QLORA_CHECKPOINT",json.dumps(final,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
