from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/kolihhan/small-agent-qlora.git"
EXPERIMENT_SHA = "f21c1e5747eaccbcccdde47a2070b21c20ac6135"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ROOT = Path("/kaggle/working")
REPO = ROOT / "small-agent-qlora"
MODEL = ROOT / "qwen3.5-4b"
RUNS = ROOT / "result"


def run(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=cwd, check=check, text=True)


def main() -> None:
    run("nvidia-smi")
    if REPO.exists():
        shutil.rmtree(REPO)
    run("git", "clone", REPO_URL, str(REPO))
    run("git", "checkout", EXPERIMENT_SHA, cwd=REPO)

    run(sys.executable, "-m", "pip", "install", "-e", ".[train,files]", cwd=REPO)
    # Qwen3.5's Gated DeltaNet training path is extremely slow without these CUDA kernels.
    run(sys.executable, "-m", "pip", "install", "-U", "flash-linear-attention>=0.4.2", "--no-build-isolation")
    run(sys.executable, "-m", "pip", "install", "-U", "git+https://github.com/Dao-AILab/causal-conv1d", "--no-build-isolation")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(MODEL),
    )

    RUNS.mkdir(parents=True, exist_ok=True)
    oracle = RUNS / "oracle"
    protected = RUNS / "empty-protected.json"
    protected.write_text(
        json.dumps({"algorithm": "sha256-normalized-question-v1", "hashes": []}),
        encoding="utf-8",
    )

    run("small-agent", "generate-oracle-policy", "--output-dir", str(oracle), "--seed", "tinyagent-policy-v1", cwd=REPO)

    common_eval = [
        "small-agent", "eval-policy",
        "--tasks", str(oracle / "test.jsonl"),
        "--backend", "transformers",
        "--hf-model", str(MODEL),
        "--max-steps", "12",
        "--max-new-tokens", "512",
    ]

    run(
        *common_eval,
        "--work-root", str(RUNS / "base-work"),
        "--output", str(RUNS / "base-policy.json"),
        cwd=REPO,
    )

    adapter = RUNS / "adapter"
    run(
        "small-agent", "train-qlora",
        "--data", str(oracle / "train.jsonl"),
        "--output", str(adapter),
        "--hf-model", str(MODEL),
        "--max-length", "1536",
        "--epochs", "2",
        "--protected-questions", str(protected),
        cwd=REPO,
    )

    run(
        *common_eval,
        "--adapter", str(adapter),
        "--work-root", str(RUNS / "adapter-work"),
        "--output", str(RUNS / "adapter-policy.json"),
        cwd=REPO,
    )

    verdict = run(
        "small-agent", "compare-policy-evals",
        "--base", str(RUNS / "base-policy.json"),
        "--candidate", str(RUNS / "adapter-policy.json"),
        "--min-exact-pp", "10.0",
        "--output", str(RUNS / "verdict.json"),
        cwd=REPO,
        check=False,
    )

    import torch

    manifest = {
        "schema_version": "tinyagent-qlora-kaggle/v1",
        "experiment_sha": EXPERIMENT_SHA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_rows": 1600,
        "heldout_rows": 200,
        "epochs": 2,
        "max_length": 1536,
        "promotion_exit_code": verdict.returncode,
    }
    (RUNS / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
