"""Fail-closed gate for automatic GAIA evaluation of a physically verified QLoRA v8.

Only verified full-state step100 from the user-approved Colab run is eligible.
No protected GAIA data or private training rows are read by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

SOURCE_SHA = "3eac23cca117a41855f4998399f79a5d9d5f4369"
TRAIN_SHA = "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
STAGE70_RUN = 44861054303
STAGE70_ARTIFACT = 14693522324
MODEL = "Qwen/Qwen3.5-9B"
PROTOCOL = "qwen9b-v8-stage70-to100-original100-scheduler-v1"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
STATE_FILES = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
SHA_FIELDS = (
    ("adapter_model.safetensors", "adapter_sha256"),
    ("optimizer.pt", "optimizer_sha256"),
    ("scheduler.pt", "scheduler_sha256"),
    ("rng_state.pth", "rng_sha256"),
)


def _digest_stream(inp) -> str:
    checksum = hashlib.sha256()
    for part in iter(lambda: inp.read(4 * 1024 * 1024), b""):
        checksum.update(part)
    return checksum.hexdigest()


def _load_json(archive: zipfile.ZipFile, name: str) -> dict:
    try:
        data = json.loads(archive.read(name))
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        raise ValueError("invalid stage100 JSON: " + name) from exc
    if not isinstance(data, dict):
        raise ValueError("invalid stage100 object: " + name)
    return data


def validate_stage100_archive(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("no physical stage100 archive")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(set(names)) != len(names):
            raise ValueError("unsafe duplicate archive entries")
        if any(
            name.startswith("/") or ".." in Path(name).parts
            or (stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK)
            for name, info in ((x.filename, x) for x in z.infolist())
        ):
            raise ValueError("unsafe archive path or symlink")
        required = {
            "stage100-verified-report.json", "run_config.json",
            *(f"checkpoint-100/{p}" for p in ADAPTER_FILES + STATE_FILES),
        }
        if not required.issubset(set(names)):
            raise ValueError("incomplete physical step100 optimizer state")
        report = _load_json(z, "stage100-verified-report.json")
        cfg = _load_json(z, "run_config.json")
        state = _load_json(z, "checkpoint-100/trainer_state.json")
        adapter_cfg = _load_json(z, "checkpoint-100/adapter_config.json")
        # Original stage70 independently had BF16 (Actions #44979601195).
        # Only this archived single-field false-negative may be evaluated;
        # never edit its original UNVERIFIED manifest or weaken SHA checks.
        audited_bf16 = (
            path.name == "qlora-v8-stage100-UNVERIFIED.zip"
            and report.get("full_v8_completed") is False
            and report.get("failed_checks") == ["compute_dtype"]
            and cfg.get("compute_dtype") == report.get("compute_dtype") == "torch.bfloat16"
        )
        conditions = {
            "protocol": report.get("protocol") == PROTOCOL,
            "full_v8_completed": report.get("full_v8_completed") is True or audited_bf16,
            "gaia_evaluated": report.get("gaia_evaluated") is False,
            "failed_checks": report.get("failed_checks") == [] or audited_bf16,
            "model": report.get("model") == MODEL,
            "stage70_source_run": report.get("stage70_source_run") == STAGE70_RUN,
            "stage70_artifact": report.get("stage70_artifact") == STAGE70_ARTIFACT,
            "original_training_code_sha": report.get("original_training_code_sha") == SOURCE_SHA,
            "train_sha256": report.get("train_sha256") == TRAIN_SHA,
            "trained_step_start": report.get("trained_step_start") == 70,
            "trained_step_end": report.get("trained_step_end") == 100,
            "scheduler_max_steps": report.get("scheduler_max_steps") == 100,
            "compute_dtype": report.get("compute_dtype") == "torch.float16" or audited_bf16,
            "global_step": state.get("global_step") == 100,
            "state_max_steps": state.get("max_steps") == 100,
            "config_model": cfg.get("model") == MODEL,
            "config_max_steps": cfg.get("max_steps") == 100,
            "config_pause_at_step": cfg.get("pause_at_step") is None,
            "config_save_steps": cfg.get("save_steps") == 10,
            "config_train_examples": cfg.get("training_examples") == 798,
            "config_compute_dtype": cfg.get("compute_dtype") == "torch.float16" or audited_bf16,
            "config_resume": cfg.get("resume_from_checkpoint") == "/content/qlora-v8-adapter/checkpoint-70",
            "adapter_base": adapter_cfg.get("base_model_name_or_path") == MODEL,
        }
        failures = [name for name, ok in conditions.items() if not ok]
        if failures:
            raise ValueError("step100 acceptance mismatch: " + ",".join(failures))
        for fname, field in SHA_FIELDS:
            if not isinstance(report.get(field), str) or len(report[field]) != 64:
                raise ValueError("missing stage100 SHA-256 for " + field)
            with z.open("checkpoint-100/" + fname) as inp:
                if _digest_stream(inp) != report[field]:
                    raise ValueError("physical step100 SHA mismatch for " + fname)
        return {**report, "evaluation_acceptance": "audited_bf16_continuity" if audited_bf16 else "original_strict"}


def extract_verified_adapter(path: str | Path, destination: str | Path) -> dict:
    """Perform all checks before extracting the two PEFT files to a fresh directory."""
    report = validate_stage100_archive(path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as z:
        for name in ADAPTER_FILES:
            target = destination / name
            with z.open("checkpoint-100/" + name) as source, target.open("xb") as output:
                for part in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(part)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract-adapter", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    if args.extract_adapter:
        report = extract_verified_adapter(args.archive, args.extract_adapter)
    else:
        report = validate_stage100_archive(args.archive)
    if args.output_report:
        args.output_report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("V8_STAGE100_PHYSICAL_STATE_VERIFIED", json.dumps({
        "trained_step_end": report["trained_step_end"],
        "model": report["model"],
        "adapter_sha256": report["adapter_sha256"],
        "source_sha": report["original_training_code_sha"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
