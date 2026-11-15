"""One-shot physical CUDA evaluation of a proven Qwen3.5-9B v8 adapter.

Runs only AFTER the separate training workflow has uploaded a verified adapter.
Uses frozen 35 independent whole parents; no training, no GAIA claim.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SOURCE_SHA = "__SOURCE_SHA__"
BRANCH = "experiment/qlora-v8-t4-preserve-frozen-half-20261115"
TRAIN_SOURCE_SHA = "6bf0b4f548781df1927ce063b7b701d2a2310719"
TRAIN_RUN_ID = 43671254880
BASE = Path("/content")
HOLDOUT = BASE / "qlora-v8-holdout.jsonl"
AUDIT = BASE / "qlora-v8-audit.json"
CHECKPOINT_ZIP = BASE / "qlora-v8-adapter.zip"
ADAPTER_DIR = BASE / "qlora-v8-verified-adapter"
FINAL = BASE / "qlora-v8-heldout-paired.json"


def checked(*command: str) -> None:
    subprocess.run(list(command), check=True)


def main() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Physical GPU is required; reject CPU substitute")
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    if vram < 14:
        raise RuntimeError("9B 4-bit model cannot be evaluated with this GPU VRAM budget")
    if SOURCE_SHA == "__SOURCE_SHA__" or len(SOURCE_SHA) != 40:
        raise RuntimeError("evaluation commit SHA not attested")
    if not HOLDOUT.is_file() or not AUDIT.is_file() or not CHECKPOINT_ZIP.is_file():
        raise FileNotFoundError("frozen heldout + audit + physically trained adapter required")
    if FINAL.exists() or ADAPTER_DIR.exists():
        raise FileExistsError("refuse to reuse stale heldout score or adapter directory")

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["train_parents"] == 304 and audit["train_actions"] == 798
    assert audit["validation_parents"] == 35 and audit["validation_actions"] == 176
    assert audit["no_gaia_training"] and audit["gaia_evaluated"] is False
    if sha256(HOLDOUT.read_bytes()).hexdigest() != audit["validation_sha256"]:
        raise ValueError("holdout sha256 mismatch: refuse evaluation")

    ADAPTER_DIR.mkdir()
    with zipfile.ZipFile(CHECKPOINT_ZIP) as archive:
        names = set(archive.namelist())
        expected = {"adapter_model.safetensors", "adapter_config.json",
                    "run_config.json", "v8-verified-report.json"}
        if not expected.issubset(names):
            raise ValueError("training artifact lacks verified adapter files")
        if any(name.startswith("/") or ".." in Path(name).parts for name in archive.namelist()):
            raise ValueError("path traversal in training artifact")
        for name in expected:
            (ADAPTER_DIR / name).write_bytes(archive.read(name))

    report = json.loads((ADAPTER_DIR/"v8-verified-report.json").read_text(encoding="utf-8"))
    if (report.get("source_sha") != TRAIN_SOURCE_SHA
        or report.get("optimizer_steps") != 100
        or report.get("adapter_initialized") != "from-base-not-v7"):
        raise ValueError("checkpoint not from the expected frozen 100-step Qwen9B run")
    weights = ADAPTER_DIR / "adapter_model.safetensors"
    if weights.stat().st_size < 1_000_000 or sha256(weights.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("physical Qwen9B LoRA safetensors missing or modified")

    repo = BASE / "qlora-v8-eval-source"
    if repo.exists():
        raise FileExistsError("refuse reused repository checkout")
    checked("git", "clone", "--branch", BRANCH, "--depth", "30",
            "https://github.com/kolihhan/small-agent-qlora.git", str(repo))
    checked("git", "-C", str(repo), "checkout", "--detach", SOURCE_SHA)
    actual = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual != SOURCE_SHA:
        raise RuntimeError("evaluation repository checkout hash drift")
    checked(sys.executable, "-m", "pip", "install", "-q",
            "-e", str(repo)+"[train,files]", "jinja2>=3.1", "sentencepiece>=0.2")
    sys.path.insert(0,str(repo / "src"))
    sys.path.insert(0,str(repo))
    from scripts.qlora_v8_holdout_paired import evaluate
    start = time.monotonic()
    outcome = evaluate(HOLDOUT,AUDIT,ADAPTER_DIR)
    outcome["evaluation_source_sha"] = SOURCE_SHA
    outcome["training_run_id"] = TRAIN_RUN_ID
    outcome["training_source_sha"] = TRAIN_SOURCE_SHA
    outcome["gpu"] = torch.cuda.get_device_name(0)
    outcome["total_duration_s"] = round(time.monotonic()-start,1)
    FINAL.write_text(json.dumps(outcome,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
    print("QLORA_V8_REAL_QWEN9B_HELDOUT",json.dumps({
        k:v for k,v in outcome["score"].items() if k!="cases"
    },ensure_ascii=False,sort_keys=True),flush=True)
    print("QLORA_V8_HELDOUT_FILE_SHA256",sha256(FINAL.read_bytes()).hexdigest(),flush=True)


if __name__=="__main__":
    main()
