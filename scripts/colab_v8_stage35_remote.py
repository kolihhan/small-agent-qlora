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
BRANCH="experiment/qlora-v8-durable-stage35-20261115"
TRAIN=Path("/content/qlora-v8-train.jsonl")
AUDIT=Path("/content/qlora-v8-audit.json")
OUT=Path("/content/qlora-v8-adapter")
ARCHIVE=Path("/content/qlora-v8-stage35-checkpoint.zip")
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
    required=("pause_at_step","data_path","output_dir","model_name","max_length",
              "epochs","learning_rate","lora_r","lora_alpha",
              "lora_dropout","max_steps","save_steps","gradient_accumulation_steps",
              "lora_target_modules","exposure_count","require_full_prompt",
              "preserve_large_half_parameters")
    missing=sorted(set(required)-set(inspect.signature(train_qlora).parameters))
    if missing:
        raise RuntimeError("QLORA_V8_TRAIN_SIGNATURE_DRIFT:"+",".join(missing))
    print("QLORA_V8_TRAIN_SIGNATURE_VERIFIED",len(required),flush=True)

    # Test the exact TRL backward GEMM primitive on the actual assigned CUDA
    # architecture BEFORE loading 9B or consuming a training step.
    from gaia_small_agent.training.qlora import _configure_trl_legacy_addmm
    from trl.kernels import chunked_logprob
    capability=torch.cuda.get_device_capability(0)
    if _configure_trl_legacy_addmm(chunked_logprob,capability=capability):
        for dtype in (torch.float16,torch.bfloat16):
            a=torch.ones((2,3),dtype=dtype,device="cuda")
            b=torch.ones((3,2),dtype=dtype,device="cuda")
            acc=torch.zeros((2,2),dtype=torch.float32,device="cuda")
            chunked_logprob._addmm_fp32(acc,a,b)
            torch.cuda.synchronize()
            assert torch.all(acc==3),f"TRL FP32 fallback mismatch: {dtype}"
        print("QLORA_T4_TRL_BACKWARD_GEMM_ACTUAL_CUDA_PREFLIGHT_PASS",
              {"compute_capability":capability,"tested":["fp16","bf16"]},flush=True)
    else:
        print("QLORA_NATIVE_TRL_GEMM_FASTPATH_SUPPORTED",capability,flush=True)

    # Crucial: keep the full 100-step optimizer scheduler despite stopping
    # once at step 35. This avoids the original uninterrupted 3h Colab loss.
    start=time.monotonic()
    train_qlora(
        data_path=TRAIN,output_dir=OUT,model_name=MODEL,
        max_length=MAX_CONTEXT,epochs=1.0,learning_rate=2e-5,
        lora_r=16,lora_alpha=32,lora_dropout=0.05,
        max_steps=MAX_STEPS,save_steps=35,gradient_accumulation_steps=8,
        lora_target_modules="all-linear",exposure_count=None,
        require_full_prompt=True,preserve_large_half_parameters=True,
        pause_at_step=35,
    )
    ckpt=OUT/"checkpoint-35"
    state=json.loads((ckpt/"trainer_state.json").read_text())
    cfg=json.loads((OUT/"run_config.json").read_text())
    if (int(state["global_step"])!=35 or cfg.get("max_steps")!=100
        or cfg.get("pause_at_step")!=35 or cfg.get("training_examples")!=798
        or cfg.get("preserve_large_half_parameters") is not True):
        raise RuntimeError("original 100-step schedule / actual stage35 proof mismatch")
    required_files=("adapter_model.safetensors","adapter_config.json","optimizer.pt",
                    "scheduler.pt","rng_state.pth","trainer_state.json")
    for name in required_files:
        if not (ckpt/name).is_file():
            raise RuntimeError(f"checkpoint-35 missing exact trainer state: {name}")
    weights=ckpt/"adapter_model.safetensors"
    if weights.stat().st_size<=1_000_000:
        raise RuntimeError("checkpoint-35 does not contain real Qwen9B LoRA weights")
    digest=sha256(weights.read_bytes()).hexdigest()
    report_stage={
        "protocol":"qwen9b-v8-partial-from-base-full-scheduler-stage35-v1",
        "model":MODEL,"source_sha":SOURCE_SHA,"train_sha256":report["train_sha256"],
        "train_actions":798,"train_parents":304,"optimizer_steps":35,
        "full_scheduler_max_steps":100,"max_context":MAX_CONTEXT,
        "checkpoint_sha256":digest,"checkpoint_bytes":weights.stat().st_size,
        "optimizer_state_sha256":sha256((ckpt/"optimizer.pt").read_bytes()).hexdigest(),
        "scheduler_state_sha256":sha256((ckpt/"scheduler.pt").read_bytes()).hexdigest(),
        "rng_state_sha256":sha256((ckpt/"rng_state.pth").read_bytes()).hexdigest(),
        "duration_s":round(time.monotonic()-start),
        "adapter_initialized":"from-base-not-v7",
        "gaia_evaluated":False,
        "external_generalization":"NOT_EVALUATED",
        "completed_full_v8":False,
        "checkpoint_resumption_requires_original_optimizer_scheduler_rng":True,
    }
    (OUT/"v8-stage35-verified-report.json").write_text(
        json.dumps(report_stage,indent=2,sort_keys=True)+"\n")
    with zipfile.ZipFile(ARCHIVE,"x",compression=zipfile.ZIP_STORED) as archive:
        for file in sorted(ckpt.rglob("*")):
            if file.is_file():
                archive.write(file,arcname=str(file.relative_to(OUT)))
        for name in ("run_config.json","v8-stage35-verified-report.json"):
            archive.write(OUT/name,arcname=name)
    print("QLORA_V8_DURABLE_STAGE35_PHYSICAL",json.dumps(report_stage,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
