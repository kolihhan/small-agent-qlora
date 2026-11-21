"""One-shot resume of original Qwen3.5-9B v8 QLoRA from verified Colab step35.

No new model/data, no GAIA supervision, no learning-rate restart. The exact
35-step PEFT weights, trainer optimizer/scheduler/RNG, and data SHA must match.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SOURCE_SHA="3eac23cca117a41855f4998399f79a5d9d5f4369"
STAGE1_WEIGHTS_SHA="3fe28aeaefe48d585c229ec56b0637542158fbdc98a29b09d8539ecee49e58c5"
STAGE1_OPTIM_SHA="617b74a3d6a86582c4bb9ed0f7770ab4c6412c59538f886e775037248c578d53"
STAGE1_SCHED_SHA="13251f4862cd831b28f88876bc62a45aac2949c43da0d5f448fb420a89e72549"
STAGE1_RNG_SHA="26f8ccba62f97c980daf96f72f5bd18d4ead0aab53022b0ba80e5b6981084f5e"
STAGE1_TRAIN_SHA="203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
MODEL="Qwen/Qwen3.5-9B"
TRAIN=Path("/content/qlora-v8-train.jsonl")
AUDIT=Path("/content/qlora-v8-audit.json")
PARTS_MANIFEST=Path("/content/stage35-manifest.json")
OUT=Path("/content/qlora-v8-adapter")
BUNDLE=Path("/content/qlora-v8-stage35-checkpoint.zip")
ARCHIVE=Path("/content/qlora-v8-stage70-checkpoint.zip")
PAUSE=70
MAX_STEPS=100


def checked(*cmd):
    subprocess.run(list(cmd),check=True)


def revive() -> dict:
    if OUT.exists() or BUNDLE.exists() or ARCHIVE.exists():
        raise FileExistsError("refuse prior checkpoint workspace reuse")
    manifest=json.loads(PARTS_MANIFEST.read_text())
    if (manifest.get("stage1_run")!=44781908112
        or manifest.get("stage1_artifact")!=14661315509
        or manifest.get("stage1_train_sha")!=STAGE1_TRAIN_SHA
        or not isinstance(manifest.get("parts"),list)
        or not 20<=len(manifest["parts"])<=60):
        raise ValueError("stage35 provenance/part count drift")
    digest=__import__("hashlib").sha256()
    total=0
    with BUNDLE.open("xb") as output:
        for entry in manifest["parts"]:
            name=entry["name"]
            if (not name.startswith("chunk-") or not name.endswith(".part")
                or name!=Path(name).name):
                raise ValueError("invalid uploaded chunk name")
            path=Path("/content")/name
            payload=path.read_bytes()
            if (len(payload)!=entry["size"] or
                sha256(payload).hexdigest()!=entry["sha256"]):
                raise ValueError("checkpoint part mismatch")
            digest.update(payload)
            output.write(payload)
            total+=len(payload)
    if (total!=manifest["size"] or
        digest.hexdigest()!=manifest["sha256"]):
        raise ValueError("whole stage35 archive checksum failure")
    with zipfile.ZipFile(BUNDLE) as archive:
        names={item.filename for item in archive.infolist()}
        needed=("checkpoint-35/adapter_model.safetensors",
                "checkpoint-35/optimizer.pt","checkpoint-35/scheduler.pt",
                "checkpoint-35/rng_state.pth","checkpoint-35/trainer_state.json",
                "run_config.json","v8-stage35-verified-report.json")
        if not set(needed).issubset(names) or any(
            name.startswith("/") or ".." in Path(name).parts for name in names
        ):
            raise ValueError("stage35 checkpoint contents invalid")
        proof=json.loads(archive.read("v8-stage35-verified-report.json"))
        cfg=json.loads(archive.read("run_config.json"))
        state=json.loads(archive.read("checkpoint-35/trainer_state.json"))
        if (proof.get("protocol")!="qwen9b-v8-partial-from-base-full-scheduler-stage35-v1"
            or proof.get("source_sha")!=SOURCE_SHA
            or proof.get("train_sha256")!=STAGE1_TRAIN_SHA
            or proof.get("checkpoint_sha256")!=STAGE1_WEIGHTS_SHA
            or proof.get("optimizer_state_sha256")!=STAGE1_OPTIM_SHA
            or proof.get("scheduler_state_sha256")!=STAGE1_SCHED_SHA
            or proof.get("rng_state_sha256")!=STAGE1_RNG_SHA
            or proof.get("optimizer_steps")!=35
            or proof.get("completed_full_v8") is not False
            or cfg.get("max_steps")!=100 or cfg.get("pause_at_step")!=35
            or cfg.get("max_length")!=3200 or cfg.get("training_examples")!=798
            or state.get("global_step")!=35 or state.get("max_steps")!=100):
            raise ValueError("original stage35 training state changed")
        for filename, expected in (
            ("checkpoint-35/adapter_model.safetensors",STAGE1_WEIGHTS_SHA),
            ("checkpoint-35/optimizer.pt",STAGE1_OPTIM_SHA),
            ("checkpoint-35/scheduler.pt",STAGE1_SCHED_SHA),
            ("checkpoint-35/rng_state.pth",STAGE1_RNG_SHA),
        ):
            if sha256(archive.read(filename)).hexdigest()!=expected:
                raise ValueError("stage35 physical optimizer/weights corruption")
        OUT.mkdir(parents=True,exist_ok=False)
        for item in archive.infolist():
            if item.filename.startswith("checkpoint-35/"):
                archive.extract(item,OUT)
    print("STAGE35_FULL_OPTIMIZER_RNG_RESTORED",flush=True)
    return proof


def main():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<14*2**30:
        raise RuntimeError("actual >=14GiB GPU required; no CPU false success")
    audit=json.loads(AUDIT.read_text())
    if (audit.get("train_sha256")!=STAGE1_TRAIN_SHA
        or sha256(TRAIN.read_bytes()).hexdigest()!=STAGE1_TRAIN_SHA
        or audit.get("training_ready_no_gpu_yet") is not True
        or audit.get("train_actions")!=798 or audit.get("train_parents")!=304
        or audit.get("validation_parents")!=35
        or audit.get("protected_question_hashes_loaded")!=165
        or audit.get("no_gaia_training") is not True):
        raise ValueError("training source/eval isolation drift")
    stage1=revive()
    repo=Path("/content/v8-frozen-original-code")
    if repo.exists():
        raise FileExistsError("reused Git repository")
    checked("git","clone","--depth","30","--branch","experiment/qlora-v8-durable-stage35-20261115",
            "https://github.com/kolihhan/small-agent-qlora.git",str(repo))
    checked("git","-C",str(repo),"checkout","--detach",SOURCE_SHA)
    actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual!=SOURCE_SHA:
        raise RuntimeError("stage35 training implementation changed")
    checked(sys.executable,"-m","pip","install","-q","-e",str(repo)+"[train,files]",
            "jinja2>=3.1","sentencepiece>=0.2")
    sys.path.insert(0,str(repo/"src"))
    from gaia_small_agent.training.qlora import train_qlora
    from gaia_small_agent.training.qlora import _configure_trl_legacy_addmm
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
    started=time.monotonic()
    print("QLORA_STAGE70_RESUME_START",json.dumps({
        "previous_step":35,"target_step":PAUSE,"model":MODEL,
        "train_parent_count":304,"train_actions":798,
        "source_sha":SOURCE_SHA,"schedule_max_steps":MAX_STEPS
    }),flush=True)
    train_qlora(
        data_path=TRAIN,output_dir=OUT,model_name=MODEL,
        max_length=3200,epochs=1.0,learning_rate=2e-5,
        lora_r=16,lora_alpha=32,lora_dropout=0.05,
        max_steps=MAX_STEPS,save_steps=35,gradient_accumulation_steps=8,
        lora_target_modules="all-linear",exposure_count=None,
        require_full_prompt=True,preserve_large_half_parameters=True,
        pause_at_step=PAUSE,
        resume_from_checkpoint=OUT/"checkpoint-35",
    )
    ckpt=OUT/f"checkpoint-{PAUSE}"
    state=json.loads((ckpt/"trainer_state.json").read_text())
    config=json.loads((OUT/"run_config.json").read_text())
    if (state.get("global_step")!=PAUSE or state.get("max_steps")!=100
        or config.get("max_steps")!=100 or config.get("pause_at_step")!=PAUSE
        or config.get("resume_from_checkpoint")!=str(OUT/"checkpoint-35")
        or config.get("training_examples")!=798):
        raise RuntimeError("stage70 model state or resume protocol drift")
    required=("adapter_model.safetensors","adapter_config.json","optimizer.pt",
              "scheduler.pt","rng_state.pth","trainer_state.json")
    if any(not (ckpt/name).is_file() for name in required):
        raise ValueError("stage70 failed to persist full optimizer checkpoint")
    def digest(name):
        return sha256((ckpt/name).read_bytes()).hexdigest()
    report={
        "protocol":"qwen9b-v8-continued-step35-to70-original100-scheduler-v1",
        "original_stage35_run":44781908112,
        "original_stage35_weights_sha":STAGE1_WEIGHTS_SHA,
        "original_training_code_sha":SOURCE_SHA,
        "model":MODEL,"train_sha256":STAGE1_TRAIN_SHA,
        "trained_step_start":35,"trained_step_end":70,"scheduler_max_steps":100,
        "adapter_sha256":digest("adapter_model.safetensors"),
        "optimizer_sha256":digest("optimizer.pt"),
        "scheduler_sha256":digest("scheduler.pt"),
        "rng_sha256":digest("rng_state.pth"),
        "duration_s":round(time.monotonic()-started),
        "full_v8_completed":False,"gaia_evaluated":False,
    }
    (OUT/"stage70-verified-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    with zipfile.ZipFile(ARCHIVE,"x",compression=zipfile.ZIP_STORED) as z:
        for item in sorted(ckpt.rglob("*")):
            if item.is_file():
                z.write(item,arcname=str(item.relative_to(OUT)))
        z.write(OUT/"run_config.json",arcname="run_config.json")
        z.write(OUT/"stage70-verified-report.json",arcname="stage70-verified-report.json")
    print("QLORA_V8_STAGE70_VERIFIED",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
