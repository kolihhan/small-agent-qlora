"""Run actual Qwen3.5-9B NF4 QLoRA (one low-LR epoch) on source-balanced
observed+synthetic TRAIN actions, never on the independent holdout/GAIA.

The trusted GitHub Actions runner uploads SHA-checked v7 JSONL files to Colab
BEFORE running this script. No credentials are transferred to the GPU VM.
Source checkout is pinned by replacing __SOURCE_SHA__ on the runner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SOURCE_SHA = "__SOURCE_SHA__"
BRANCH = "experiment/qlora-v7-source-balanced-sft-20261113"
MODEL = "Qwen/Qwen3.5-9B"
TRAIN = Path("/content/qlora-v7-train.jsonl")
HOLDOUT = Path("/content/qlora-v7-heldout.jsonl")
AUDIT = Path("/content/qlora-v7-audit.json")
OUT = Path("/content/qlora-v7-adapter")
ARCHIVE = Path("/content/qlora-v7-adapter.zip")


def checked(*args):
    print("STEP",args[0],args[1] if len(args)>1 else "",flush=True)
    subprocess.run(list(args),check=True)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not allocated: never pretend CPU QLoRA is equivalent")
    vram=torch.cuda.get_device_properties(0).total_memory/(1024**3)
    if vram<14.5:
        raise RuntimeError("Qwen9B NF4 requires >=14.5 GiB")
    if SOURCE_SHA=="__SOURCE_SHA__" or len(SOURCE_SHA)!=40:
        raise RuntimeError("source SHA not pinned by trusted runner")
    expected=json.loads(AUDIT.read_text(encoding="utf-8"))
    if expected.get("schema")!="qlora-v7-whole-task-source-balanced-20261113-v1":
        raise ValueError("wrong v7 source protocol")
    if expected["train_tasks"]<200 or expected["train_actions"]<500:
        raise ValueError("v7 dataset unexpectedly small")
    if expected["train_real_action_fraction"]<0.30:
        raise ValueError("insufficient real observed tool behavior")
    for path,key in ((TRAIN,"train_sha256"),(HOLDOUT,"holdout_sha256")):
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:
            raise ValueError("training or holdout source SHA mismatch")
    print("V7_GPU_SOURCE_VERIFIED",json.dumps({
        "tasks":expected["train_tasks"],
        "train_actions":expected["train_actions"],
        "source_families":expected["train_source_counts"],
        "gpu":torch.cuda.get_device_name(0),"vram_gib":round(vram,2),
    }),flush=True)
    repo=Path("/content/small-agent-qlora-v7")
    if repo.exists() or OUT.exists() or ARCHIVE.exists():
        raise FileExistsError("refuse to reuse stale Colab model workspace")
    checked("git","clone","--branch",BRANCH,"--depth","100",
            "https://github.com/kolihhan/small-agent-qlora.git",str(repo))
    checked("git","-C",str(repo),"checkout","--detach",SOURCE_SHA)
    actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual != SOURCE_SHA:
        raise ValueError("source SHA drift")
    checked(sys.executable,"-m","pip","install","-q","-e",str(repo)+"[train,dev,files]",
            "jinja2>=3.1")
    sys.path.insert(0,str(repo/"src"))
    sys.path.insert(0,str(repo))
    from gaia_small_agent.training.qlora import train_qlora
    # Only 1 epoch-worth optimizer budget; do not repeatedly memorize the
    # same 44 real parent tasks. The 11 observed held-out parents never train.
    steps=65
    started=time.time()
    train_qlora(
        data_path=TRAIN, output_dir=OUT, model_name=MODEL,
        max_length=1536,epochs=1.0,learning_rate=5e-5,
        lora_r=16,lora_alpha=32,lora_dropout=0.05,
        max_steps=steps,save_steps=25,
        gradient_accumulation_steps=8,
        lora_target_modules="all-linear",exposure_count=None,
        require_full_prompt=True,
        preserve_large_half_parameters=(vram<20),
    )
    weights=OUT/"adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size<=1000000:
        raise RuntimeError("missing physical LoRA weights after training")
    cfg=json.loads((OUT/"run_config.json").read_text(encoding="utf-8"))
    if cfg.get("max_steps")!=steps or cfg.get("training_examples")!=expected["train_actions"]:
        raise ValueError("trained unexpected number of steps or examples")
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    report={
        "schema":"qlora-v7-actual-nf4-sft-checkpoint-v1",
        "source_sha":SOURCE_SHA,"model":MODEL,
        "adapter_sha256":digest,"adapter_size":weights.stat().st_size,
        "optimizer_steps_target":steps,
        "train_actions":expected["train_actions"],
        "train_parents":expected["train_tasks"],
        "heldout_parents_unseen":expected["holdout_tasks"],
        "real_fraction":expected["train_real_action_fraction"],
        "elapsed_s":round(time.time()-started),
        "gaia_accuracy":"NOT_EVALUATED",
        "external_generalization":"UNVERIFIED",
    }
    (OUT/"verified-v7-report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    # Preserve verified checkpoint before any evaluation, so compute failures
    # never create a fake success signal.
    with zipfile.ZipFile(ARCHIVE,"x",compression=zipfile.ZIP_STORED) as z:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                z.write(path,arcname=path.name)
    print("QLORA_V7_TRAINED_ADAPTER_VERIFIED",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
