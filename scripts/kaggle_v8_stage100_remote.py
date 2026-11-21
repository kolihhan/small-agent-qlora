"""Kaggle T4 continuation of the *original* Qwen3.5-9B v8 QLoRA steps 70→100.

Loads immutable privately attached Kaggle Dataset (original verified GitHub
artifact + frozen 798-action training rows). Recovers actual optimizer/scheduler/
RNG, and resumes the exact original 100-step scheduler and trainer revision.
No Kaggle competitions or GAIA labels are used in training.
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

SOURCE_SHA = "3eac23cca117a41855f4998399f79a5d9d5f4369"
TRAIN_SHA = "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
MODEL = "Qwen/Qwen3.5-9B"
STAGE70_RUN = 44861054303
STAGE70_ARTIFACT = 14693522324
STAGE70_FILE_SIZE = 539885434
STAGE70_WEIGHTS_SHA = "214ca477a95c8dd9c4395d1093ada147eb7fed62d33b3781f846d28d54f4c8c8"
STAGE70_OPTIM_SHA = "d5e928f8029d99f4dd94e805a65a7b33973dfe04ed407da9a81154053ec6ae99"
STAGE70_SCHED_SHA = "6022b1ed992ffdf30d48cb9476b12819081d7b2f69813a50222fc783ae361512"
STAGE70_RNG_SHA = "aa78619ac2c20eff4b2694524a8b7962d5b5d6cb7b21edd426f38a2a0ec54c19"
DATASET_ID = "kolihhan/qlora-v8-stage70-verified-20261121"
DATASET_ROOT = Path("/kaggle/input/qlora-v8-stage70-verified-20261121")
# Kaggle retains only /kaggle/working output. Cache and intermediates stay in temp.
TEMP = Path("/kaggle/temp")
TRAIN = DATASET_ROOT / "qlora-v8-train.jsonl"
AUDIT = DATASET_ROOT / "qlora-v8-audit.json"
BUNDLE = DATASET_ROOT / "qlora-v8-stage70-checkpoint.zip"
SOURCE_MANIFEST = DATASET_ROOT / "stage70-kaggle-transport-manifest.json"
OUT = TEMP / "qlora-v8-adapter"
ARCHIVE = Path("/kaggle/working/qlora-v8-stage100-checkpoint.zip")
MAX_STEPS = 100


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def checked(*cmd: str) -> None:
    subprocess.run(list(cmd),check=True)


def verify_and_restore() -> dict:
    """Restore a hash-verified stage70 from ZIP OR Kaggle's auto-extracted files.

    Kaggle unpacks uploaded dataset ZIPs; the archive itself is not mounted.
    Never relax the four source tensor hashes or original train-data provenance.
    """
    import shutil

    if OUT.exists() or ARCHIVE.exists():
        raise FileExistsError("refuse reused Kaggle checkpoint workspace")
    if not all(p.is_file() for p in (TRAIN, AUDIT, SOURCE_MANIFEST)):
        raise FileNotFoundError("private Kaggle stage70 dataset is incomplete")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if (manifest.get("protocol") != "private-kaggle-original-qlora-v8-step70-v1"
        or manifest.get("stage70_artifact") != STAGE70_ARTIFACT
        or manifest.get("stage70_source_run") != STAGE70_RUN
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("stage70_zip_bytes") != STAGE70_FILE_SIZE
        or not isinstance(manifest.get("stage70_zip_sha256"), str)
        or len(manifest["stage70_zip_sha256"]) != 64):
        raise ValueError("transport source identity mismatch")

    audit = json.loads(AUDIT.read_text())
    if (sha_file(TRAIN) != TRAIN_SHA
        or audit.get("train_sha256") != TRAIN_SHA
        or audit.get("train_actions") != 798
        or audit.get("train_parents") != 304
        or audit.get("validation_parents") != 35
        or audit.get("protected_question_hashes_loaded") != 165
        or audit.get("training_ready_no_gpu_yet") is not True
        or audit.get("no_gaia_training") is not True):
        raise ValueError("frozen training data or protected GAIA source drift")

    required = {
        "checkpoint-70/adapter_model.safetensors",
        "checkpoint-70/adapter_config.json",
        "checkpoint-70/optimizer.pt",
        "checkpoint-70/scheduler.pt",
        "checkpoint-70/rng_state.pth",
        "checkpoint-70/trainer_state.json",
        "run_config.json",
        "stage70-verified-report.json",
    }
    if BUNDLE.is_file():
        if BUNDLE.stat().st_size != STAGE70_FILE_SIZE or sha_file(BUNDLE) != manifest["stage70_zip_sha256"]:
            raise ValueError("transport ZIP bytes/hash mismatch")
        with zipfile.ZipFile(BUNDLE) as z:
            names = set(z.namelist())
            if not required.issubset(names) or any(
                name.startswith("/") or ".." in Path(name).parts for name in names
            ):
                raise ValueError("archive missing optimizer state or unsafe paths")
            proof = json.loads(z.read("stage70-verified-report.json"))
            cfg = json.loads(z.read("run_config.json"))
            state = json.loads(z.read("checkpoint-70/trainer_state.json"))
            tensor_hashes = {
                name: hashlib.sha256(z.read("checkpoint-70/" + name)).hexdigest()
                for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth")
            }
            # Original names are constant and validated; ZIP extraction never traverses unknown members.
            OUT.mkdir(parents=True, exist_ok=False)
            for name in sorted(required):
                if not name.startswith("checkpoint-70/"):
                    continue
                target = OUT / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest, length=4 * 1024 * 1024)
        mode = "original_zip"
    else:
        # Kaggle dataset uploads automatically extract ZIP. Locate only a unique
        # original checkpoint-70 directory (preserving the ZIP's internal tree).
        candidates = sorted({
            state_path.parent for state_path in DATASET_ROOT.rglob("checkpoint-70/trainer_state.json")
            if state_path.is_file() and not state_path.is_symlink()
        })
        if len(candidates) != 1:
            raise FileNotFoundError("private Kaggle stage70 dataset is incomplete or ambiguous")
        source_checkpoint = candidates[0]
        source_root = source_checkpoint.parent
        if source_checkpoint.name != "checkpoint-70":
            raise ValueError("unpacked checkpoint path is not stage70")
        sources = {
            name: source_root / name for name in required
        }
        if any(not p.is_file() or p.is_symlink() for p in sources.values()):
            raise FileNotFoundError("private Kaggle stage70 dataset is incomplete")
        proof = json.loads(sources["stage70-verified-report.json"].read_text())
        cfg = json.loads(sources["run_config.json"].read_text())
        state = json.loads(sources["checkpoint-70/trainer_state.json"].read_text())
        tensor_hashes = {
            name: sha_file(sources["checkpoint-70/" + name])
            for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth")
        }
        mode = "kaggle_auto_extracted"

    expected = (
        proof.get("protocol") == "qwen9b-v8-continued-step35-to70-original100-scheduler-v1",
        proof.get("model") == MODEL,
        proof.get("original_training_code_sha") == SOURCE_SHA,
        proof.get("train_sha256") == TRAIN_SHA,
        proof.get("original_stage35_run") == 44781908112,
        proof.get("trained_step_start") == 35,
        proof.get("trained_step_end") == 70,
        proof.get("scheduler_max_steps") == MAX_STEPS,
        proof.get("full_v8_completed") is False,
        proof.get("gaia_evaluated") is False,
        cfg.get("max_steps") == MAX_STEPS,
        cfg.get("pause_at_step") == 70,
        cfg.get("max_length") == 3200,
        cfg.get("training_examples") == 798,
        state.get("global_step") == 70,
        state.get("max_steps") == MAX_STEPS,
    )
    if not all(expected):
        raise ValueError("original verified 70-step full-state training contract drift")

    hashes = (
        ("adapter_model.safetensors", STAGE70_WEIGHTS_SHA),
        ("optimizer.pt", STAGE70_OPTIM_SHA),
        ("scheduler.pt", STAGE70_SCHED_SHA),
        ("rng_state.pth", STAGE70_RNG_SHA),
    )
    for name, expected_hash in hashes:
        if tensor_hashes[name] != expected_hash:
            raise ValueError("actual stage70 tensor state SHA mismatch for " + name)

    if mode == "kaggle_auto_extracted":
        OUT.mkdir(parents=True, exist_ok=False)
        dest = OUT / "checkpoint-70"
        dest.mkdir(parents=True, exist_ok=False)
        for name in sorted(required):
            if name.startswith("checkpoint-70/"):
                source = sources[name]
                with source.open("rb") as inp, (dest / Path(name).name).open("xb") as out:
                    shutil.copyfileobj(inp, out, length=4 * 1024 * 1024)
    # Defense-in-depth: compare restored tensors, not only read-only dataset sources.
    for name, expected_hash in hashes:
        if sha_file(OUT / "checkpoint-70" / name) != expected_hash:
            raise ValueError("restored stage70 tensor state hash mismatch: " + name)
    print("KAGGLE_STAGE70_SOURCE_OPTIMIZER_SCHEDULER_RNG_INTACT",
          json.dumps({"input_mode": mode, "global_step": 70, "hashes_verified": 4}), flush=True)
    return proof


def main() -> None:
    # Kaggle GPU T4 ×2: force exactly one card for consistency with frozen Colab T4.
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("must set CUDA_VISIBLE_DEVICES=0 BEFORE Python imports torch")
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1
        or torch.cuda.get_device_capability(0)!=(7,5)
        or "T4" not in torch.cuda.get_device_name(0).upper()
        or torch.cuda.get_device_properties(0).total_memory<14*2**30):
        raise RuntimeError("requires one visible Tesla T4 with 14+GiB VRAM")
    TEMP.mkdir(parents=True,exist_ok=True)
    # HF cache outside Kaggle saved output, avoids ~18 GiB model weights in /kaggle/working.
    os.environ["HF_HOME"]=str(TEMP/"huggingface")
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    if not (os.environ.get("HF_HUB_OFFLINE") in (None,"","0")):
        raise RuntimeError("Kaggle must have Internet ON to fetch original Qwen3.5-9B")
    proof=verify_and_restore()

    repo=TEMP/"v8-frozen-original-code"
    if repo.exists():
        raise FileExistsError("original frozen source workspace reused")
    checked("git","clone","--depth","30","--branch","experiment/qlora-v8-durable-stage35-20261115",
            "https://github.com/kolihhan/small-agent-qlora.git",str(repo))
    checked("git","-C",str(repo),"checkout","--detach",SOURCE_SHA)
    actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual!=SOURCE_SHA:
        raise RuntimeError("original trainer commit changed")
    checked(sys.executable,"-m","pip","install","-q","--no-cache-dir",
            "-e",str(repo)+"[train,files]","jinja2>=3.1","sentencepiece>=0.2")
    sys.path.insert(0,str(repo/"src"))
    from gaia_small_agent.training.qlora import train_qlora,_configure_trl_legacy_addmm
    from trl.kernels import chunked_logprob
    cap=torch.cuda.get_device_capability(0)
    if _configure_trl_legacy_addmm(chunked_logprob,capability=cap):
        for dtype in (torch.float16,torch.bfloat16):
            a=torch.ones((2,3),dtype=dtype,device="cuda")
            b=torch.ones((3,2),dtype=dtype,device="cuda")
            buf=torch.zeros((2,2),dtype=torch.float32,device="cuda")
            chunked_logprob._addmm_fp32(buf,a,b)
            torch.cuda.synchronize()
            assert torch.all(buf==3)
    start=time.monotonic()
    print("KAGGLE_ORIGINAL_QWEN9B_V8_STEP70_TO100_START",
          json.dumps({"model":MODEL,"start":70,"end":100,"code_sha":SOURCE_SHA,
                      "train_sha256":TRAIN_SHA,"gpu":torch.cuda.get_device_name(0)}),
          flush=True)
    # Original 100-step optimizer/scheduler, 798 actions, same fp16 T4 and loss.
    # Only saving interval changes to preserve a complete full-state step100 checkpoint.
    train_qlora(
        data_path=TRAIN,output_dir=OUT,model_name=MODEL,
        max_length=3200,epochs=1.0,learning_rate=2e-5,
        lora_r=16,lora_alpha=32,lora_dropout=0.05,
        max_steps=MAX_STEPS,save_steps=10,gradient_accumulation_steps=8,
        lora_target_modules="all-linear",exposure_count=None,
        require_full_prompt=True,preserve_large_half_parameters=True,
        pause_at_step=None,resume_from_checkpoint=OUT/"checkpoint-70",
    )
    checkpoint=OUT/"checkpoint-100"
    state_path=checkpoint/"trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError("NO_CHECKPOINT100_CANNOT_CLAIM_COMPLETED")
    state=json.loads(state_path.read_text())
    cfg=json.loads((OUT/"run_config.json").read_text())
    if (state.get("global_step")!=100 or state.get("max_steps")!=100
        or cfg.get("max_steps")!=100 or cfg.get("pause_at_step") is not None
        or cfg.get("save_steps")!=10 or cfg.get("training_examples")!=798
        or cfg.get("resume_from_checkpoint")!=str((OUT/"checkpoint-70").resolve())
        or cfg.get("compute_dtype")!="torch.float16"):
        raise RuntimeError("final optimizer-step count, scheduler, or precision drift")
    required=("adapter_model.safetensors","adapter_config.json","optimizer.pt",
              "scheduler.pt","rng_state.pth","trainer_state.json")
    if any(not (checkpoint/p).is_file() for p in required):
        raise RuntimeError("incomplete checkpoint100 full-state")
    report={
        "protocol":"qwen9b-v8-stage70-to100-original100-scheduler-kaggle-t4-v1",
        "model":MODEL,"stage70_source_run":STAGE70_RUN,
        "stage70_artifact":STAGE70_ARTIFACT,
        "stage70_adapter_sha256":STAGE70_WEIGHTS_SHA,
        "original_training_code_sha":SOURCE_SHA,
        "train_sha256":TRAIN_SHA,"trained_step_start":70,"trained_step_end":100,
        "scheduler_max_steps":MAX_STEPS,"gaia_evaluated":False,
        "full_v8_completed":True,"gpu_name":torch.cuda.get_device_name(0),
        "compute_dtype":"torch.float16",
        "adapter_sha256":sha_file(checkpoint/"adapter_model.safetensors"),
        "optimizer_sha256":sha_file(checkpoint/"optimizer.pt"),
        "scheduler_sha256":sha_file(checkpoint/"scheduler.pt"),
        "rng_sha256":sha_file(checkpoint/"rng_state.pth"),
        "duration_s":round(time.monotonic()-start,2),
    }
    (OUT/"stage100-verified-report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    with zipfile.ZipFile(ARCHIVE,"x",compression=zipfile.ZIP_STORED) as z:
        for item in sorted(checkpoint.rglob("*")):
            if item.is_file():
                z.write(item,arcname=str(item.relative_to(OUT)))
        z.write(OUT/"run_config.json",arcname="run_config.json")
        z.write(OUT/"stage100-verified-report.json",arcname="stage100-verified-report.json")
    print("KAGGLE_ORIGINAL_QWEN9B_V8_STAGE100_FULL_STATE_VERIFIED",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
