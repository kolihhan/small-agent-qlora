"""Headless Google Colab CLI job: train full synthetic Qwen3.5-9B agent QLoRA.

This program runs ON the allocated Colab GPU VM. The official Colab CLI
transports this file from a trusted GitHub Actions runner after validating ADC.
It never reads Google credentials, asks for Drive consent, or uses GAIA data.

Source revision and data derivation are pinned. No success claim without
a physically present, validated PEFT adapter.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SOURCE_SHA = "6dd363c2c71673578d5a183162acfb0020eb3ac4"
SOURCE_BRANCH = "fix/qlora-v4-t4-4bit-memory-20261112"
MODEL = "Qwen/Qwen3.5-9B"
OUT = Path("/content/qlora-full-v4-remote")
ARCHIVE = Path("/content/qlora-full-v4-adapter.zip")


def run(*args: str) -> None:
    print("RUN", args[0], args[1] if len(args) > 1 else "", flush=True)
    subprocess.run(list(args), check=True)


def main() -> None:
    import torch
    assert torch.cuda.is_available(), "Colab CLI did not assign a CUDA GPU"
    capacity = torch.cuda.get_device_properties(0).total_memory / 2**30
    if capacity < 14.5:
        raise RuntimeError("Need >=14.5 GiB for Qwen3.5-9B NF4")
    max_length = 1024 if capacity < 20 else 2048
    print("COLAB_GPU", json.dumps({
        "device": torch.cuda.get_device_name(0),
        "vram_gib": round(capacity, 2), "max_length": max_length,
        "gpu_required": True,
    }), flush=True)

    repo = Path("/content/small-agent-qlora")
    if repo.exists():
        raise FileExistsError("Colab VM contains prior unverified source")
    run("git","clone","--branch",SOURCE_BRANCH,"--depth","100",
        "https://github.com/kolihhan/small-agent-qlora.git",str(repo))
    run("git","-C",str(repo),"checkout","--detach",SOURCE_SHA)
    actual = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    assert actual == SOURCE_SHA, "Repository revision mismatch"
    run(sys.executable,"-m","pip","install","-q","-e",
        str(repo)+"[train,dev,files]","jinja2>=3.1")
    # IPython/Colab kernels do not reload editable-install .pth files mid-session.
    # This repository uses setuptools src/ layout: explicitly expose it now.
    sys.path.insert(0,str(repo / "src"))
    sys.path.insert(0,str(repo))
    assert (repo / "src/gaia_small_agent/__init__.py").is_file()

    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    from gaia_small_agent.training.qlora import (
        _render_turn_examples, _tokenize_completion_examples, train_qlora,
    )
    from scripts.prepare_full_qlora_v4 import prepare_full_clean_train
    from transformers import AutoTokenizer

    paths = generate_oracle_policy_v3_dataset(
        Path("/content/policy-v3"),seed="small-agent-policy-v3")
    original = [
        json.loads(x) for x in paths["train"].read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    cleaned, report = prepare_full_clean_train(original)
    assert len(cleaned) == 1600
    assert report["total_trainable_actions"] == 3200
    assert report["removed_intentionally_bad_actions"] == 400
    tokenizer = AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    rendered = _render_turn_examples(tokenizer,cleaned)
    tokens = _tokenize_completion_examples(tokenizer,rendered,
                                            max_length=max_length,require_full_prompt=True)
    assert len(tokens) == 3200
    print("GPU_PREFLIGHT",json.dumps({
        "source_sha": SOURCE_SHA, "tasks":len(cleaned),
        "clean_actions":len(tokens),"bad_first_actions_removed":400,
        "longest_tokens":max(len(x["input_ids"]) for x in tokens),
    }),flush=True)
    OUT.mkdir(parents=True,exist_ok=False)
    data = Path("/content/qlora-v4-training.jsonl")
    data.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in cleaned),
                    encoding="utf-8")
    report["source_sha"] = SOURCE_SHA
    report["model"] = MODEL
    report["max_length"] = max_length
    report["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    (OUT/"data-audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    started = time.time()
    train_qlora(
        data_path=data, output_dir=OUT, model_name=MODEL,
        max_length=max_length,epochs=1.0,learning_rate=1e-4,
        lora_r=16,lora_alpha=32,lora_dropout=0.05,
        max_steps=400,save_steps=50,gradient_accumulation_steps=8,
        lora_target_modules="all-linear",exposure_count=None,
        require_full_prompt=True,
        preserve_large_half_parameters=(capacity < 20),
    )
    weights = OUT/"adapter_model.safetensors"
    assert weights.exists() and weights.stat().st_size > 1_000_000
    cfg = json.loads((OUT/"run_config.json").read_text())
    assert cfg["training_examples"] == 3200
    assert cfg["max_steps"] == 400
    assert cfg["lora_target_modules"] == "all-linear"
    assert cfg["max_length"] == max_length
    assert cfg["require_full_prompt"] is True
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    (OUT/"verified-adapter-report.json").write_text(json.dumps({
        "model":MODEL, "source_sha":SOURCE_SHA, "adapter_sha256":sha,
        "adapter_bytes":weights.stat().st_size,"optimizer_steps":400,
        "seconds":round(time.time()-started,1),
        "caveat":"No GAIA or HotpotQA result has been measured.",
    },indent=2,sort_keys=True)+"\n")
    if ARCHIVE.exists():
        raise FileExistsError("Archive already exists: refusing overwrite")
    with zipfile.ZipFile(ARCHIVE,"w",compression=zipfile.ZIP_STORED) as archive:
        for f in sorted(OUT.glob("*")):
            if f.is_file() and not f.name.startswith("."):
                archive.write(f,arcname=f.name)
    print("COLAB_V4_TRAINED_ADAPTER",json.dumps({
        "zip":str(ARCHIVE),"bytes":ARCHIVE.stat().st_size,
        "sha256":hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "gaia_score":"unmeasured",
    }),flush=True)


if __name__ == "__main__":
    main()
