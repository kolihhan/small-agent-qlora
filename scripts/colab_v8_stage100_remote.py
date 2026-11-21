"""Complete the ORIGINAL Qwen3.5-9B QLoRA v8 100-step schedule from verified step70.

Continuation retains training data SHA, optimizer/scheduler/RNG and original
training implementation. No GAIA labels or protected benchmark data in training.
The 100th checkpoint is full-state; no score claim until independent GAIA eval.
"""
from __future__ import annotations

from hashlib import sha256
import json
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

TRAIN = Path("/content/qlora-v8-train.jsonl")
AUDIT = Path("/content/qlora-v8-audit.json")
PARTS_MANIFEST = Path("/content/stage70-manifest.json")
OUT = Path("/content/qlora-v8-adapter")
BUNDLE = Path("/content/qlora-v8-stage70-checkpoint.zip")
ARCHIVE = Path("/content/qlora-v8-stage100-checkpoint.zip")
MAX_STEPS = 100


def checked(*cmd: str) -> None:
    subprocess.run(list(cmd), check=True)


def restore_verified_step70() -> dict:
    if OUT.exists() or BUNDLE.exists() or ARCHIVE.exists():
        raise FileExistsError("refuse reused checkpoint workspace")
    manifest = json.loads(PARTS_MANIFEST.read_text())
    if (manifest.get("stage70_run") != STAGE70_RUN
        or manifest.get("stage70_artifact") != STAGE70_ARTIFACT
        or manifest.get("train_sha256") != TRAIN_SHA
        or manifest.get("size") != STAGE70_FILE_SIZE
        or not isinstance(manifest.get("parts"), list)
        or not 55 <= len(manifest["parts"]) <= 90):
        raise ValueError("source checkpoint provenance or chunk count drift")
    digest = __import__("hashlib").sha256()
    length = 0
    with BUNDLE.open("xb") as out:
        for chunk in manifest["parts"]:
            name = chunk["name"]
            if (name != Path(name).name or not name.startswith("chunk-")
                or not name.endswith(".part")):
                raise ValueError("unsafe checkpoint chunk filename")
            payload = (Path("/content") / name).read_bytes()
            if (len(payload) != chunk["size"]
                or sha256(payload).hexdigest() != chunk["sha256"]):
                raise ValueError("checkpoint chunk corrupted")
            digest.update(payload)
            out.write(payload)
            length += len(payload)
    if length != manifest["size"] or digest.hexdigest() != manifest["sha256"]:
        raise ValueError("checkpoint archive digest differs")
    with zipfile.ZipFile(BUNDLE) as z:
        names = {x.filename for x in z.infolist()}
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
        if not required.issubset(names) or any(
            p.startswith("/") or ".." in Path(p).parts for p in names
        ):
            raise ValueError("stage70 archive missing full state or unsafe paths")
        proof = json.loads(z.read("stage70-verified-report.json"))
        cfg = json.loads(z.read("run_config.json"))
        state = json.loads(z.read("checkpoint-70/trainer_state.json"))
        expected = (
            (proof.get("protocol") == "qwen9b-v8-continued-step35-to70-original100-scheduler-v1"),
            (proof.get("model") == MODEL),
            (proof.get("original_training_code_sha") == SOURCE_SHA),
            (proof.get("train_sha256") == TRAIN_SHA),
            (proof.get("original_stage35_run") == 44781908112),
            (proof.get("trained_step_start") == 35),
            (proof.get("trained_step_end") == 70),
            (proof.get("scheduler_max_steps") == MAX_STEPS),
            (proof.get("full_v8_completed") is False),
            (proof.get("gaia_evaluated") is False),
            (cfg.get("max_steps") == MAX_STEPS),
            (cfg.get("pause_at_step") == 70),
            (cfg.get("max_length") == 3200),
            (cfg.get("training_examples") == 798),
            (state.get("global_step") == 70),
            (state.get("max_steps") == MAX_STEPS),
        )
        if not all(expected):
            raise ValueError("stage70 optimizer state/protocol mismatch")
        for path, target in (
            ("checkpoint-70/adapter_model.safetensors", STAGE70_WEIGHTS_SHA),
            ("checkpoint-70/optimizer.pt", STAGE70_OPTIM_SHA),
            ("checkpoint-70/scheduler.pt", STAGE70_SCHED_SHA),
            ("checkpoint-70/rng_state.pth", STAGE70_RNG_SHA),
        ):
            if sha256(z.read(path)).hexdigest() != target:
                raise ValueError("actual stage70 tensor state differs from verified log")
        OUT.mkdir(parents=True, exist_ok=False)
        for item in z.infolist():
            if item.filename.startswith("checkpoint-70/"):
                z.extract(item, OUT)
    print("STAGE70_FULL_STATE_INTACT_AND_RESTORED", flush=True)
    return proof


def main() -> None:
    import torch
    if (not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory < 14 * 2**30):
        raise RuntimeError("requires actual >=14GiB CUDA GPU")
    audit = json.loads(AUDIT.read_text())
    if (sha256(TRAIN.read_bytes()).hexdigest() != TRAIN_SHA
        or audit.get("train_sha256") != TRAIN_SHA
        or audit.get("training_ready_no_gpu_yet") is not True
        or audit.get("train_actions") != 798
        or audit.get("train_parents") != 304
        or audit.get("validation_parents") != 35
        or audit.get("protected_question_hashes_loaded") != 165
        or audit.get("no_gaia_training") is not True):
        raise ValueError("training data or protected-GAIA isolation drift")
    restore_verified_step70()

    repo = Path("/content/v8-frozen-original-code")
    if repo.exists():
        raise FileExistsError("git workspace reused")
    checked("git", "clone", "--depth", "30",
            "--branch", "experiment/qlora-v8-durable-stage35-20261115",
            "https://github.com/kolihhan/small-agent-qlora.git", str(repo))
    checked("git", "-C", str(repo), "checkout", "--detach", SOURCE_SHA)
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != SOURCE_SHA:
        raise RuntimeError("original frozen training code changed")
    checked(sys.executable, "-m", "pip", "install", "-q",
            "-e", str(repo) + "[train,files]",
            "jinja2>=3.1", "sentencepiece>=0.2")
    sys.path.insert(0, str(repo / "src"))
    from gaia_small_agent.training.qlora import (
        train_qlora, _configure_trl_legacy_addmm,
    )
    from trl.kernels import chunked_logprob
    cap = torch.cuda.get_device_capability(0)
    if _configure_trl_legacy_addmm(chunked_logprob, capability=cap):
        for dtype in (torch.float16, torch.bfloat16):
            a = torch.ones((2,3), dtype=dtype, device="cuda")
            b = torch.ones((3,2), dtype=dtype, device="cuda")
            buf = torch.zeros((2,2), dtype=torch.float32, device="cuda")
            chunked_logprob._addmm_fp32(buf, a, b)
            torch.cuda.synchronize()
            assert torch.all(buf == 3)

    started = time.monotonic()
    print("QLORA_V8_FINAL30_RESUME", json.dumps({
        "start_step": 70, "target_step": 100, "model": MODEL,
        "source_sha": SOURCE_SHA, "train_sha256": TRAIN_SHA,
        "max_steps": MAX_STEPS,
    }), flush=True)

    # Pause must be None at max_steps=100 (frozen trainer validates pause < max).
    # Save every 10 steps solely to guarantee durable optimizer state at 100.
    train_qlora(
        data_path=TRAIN, output_dir=OUT, model_name=MODEL,
        max_length=3200, epochs=1.0, learning_rate=2e-5,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        max_steps=MAX_STEPS, save_steps=10, gradient_accumulation_steps=8,
        lora_target_modules="all-linear", exposure_count=None,
        require_full_prompt=True, preserve_large_half_parameters=True,
        pause_at_step=None, resume_from_checkpoint=OUT / "checkpoint-70",
    )

    checkpoint = OUT / "checkpoint-100"
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        raise RuntimeError("final checkpoint-100 absent; cannot claim completed training")
    state = json.loads(state_path.read_text())
    cfg = json.loads((OUT / "run_config.json").read_text())
    if (state.get("global_step") != 100 or state.get("max_steps") != 100
        or cfg.get("max_steps") != 100 or cfg.get("pause_at_step") is not None
        or cfg.get("save_steps") != 10 or cfg.get("training_examples") != 798
        or cfg.get("resume_from_checkpoint") != str(OUT / "checkpoint-70")):
        raise RuntimeError("final optimizer-step count or continuation config drift")
    required = (
        "adapter_model.safetensors", "adapter_config.json",
        "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
    )
    if any(not (checkpoint / p).is_file() for p in required):
        raise RuntimeError("final full-state checkpoint incomplete")
    def digest(name: str) -> str:
        return sha256((checkpoint / name).read_bytes()).hexdigest()

    report = {
        "protocol": "qwen9b-v8-stage70-to100-original100-scheduler-v1",
        "stage70_source_run": STAGE70_RUN,
        "stage70_artifact": STAGE70_ARTIFACT,
        "stage70_adapter_sha256": STAGE70_WEIGHTS_SHA,
        "original_training_code_sha": SOURCE_SHA,
        "model": MODEL,
        "train_sha256": TRAIN_SHA,
        "trained_step_start": 70,
        "trained_step_end": 100,
        "scheduler_max_steps": MAX_STEPS,
        "adapter_sha256": digest("adapter_model.safetensors"),
        "optimizer_sha256": digest("optimizer.pt"),
        "scheduler_sha256": digest("scheduler.pt"),
        "rng_sha256": digest("rng_state.pth"),
        "duration_s": round(time.monotonic() - started),
        "full_v8_completed": True,
        "gaia_evaluated": False,
    }
    (OUT / "stage100-verified-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_STORED) as z:
        for item in sorted(checkpoint.rglob("*")):
            if item.is_file():
                z.write(item, arcname=str(item.relative_to(OUT)))
        z.write(OUT / "run_config.json", arcname="run_config.json")
        z.write(OUT / "stage100-verified-report.json",
                arcname="stage100-verified-report.json")
    print("QLORA_V8_STAGE100_VERIFIED", json.dumps(report,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
