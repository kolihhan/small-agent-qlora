"""CPU-only acceptance tests for the automatic GAIA100 checkpoint gate."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from scripts.gaia_v8_stage100_gate import validate_stage100_archive, extract_verified_adapter


def _archive(tmp_path, *, overrides=None, missing=None, damaged=None):
    config = {
        "model": "Qwen/Qwen3.5-9B",
        "max_steps": 100,
        "pause_at_step": None,
        "save_steps": 10,
        "training_examples": 798,
        "compute_dtype": "torch.float16",
        "resume_from_checkpoint": "/content/qlora-v8-adapter/checkpoint-70",
    }
    state = {"global_step": 100, "max_steps": 100}
    raw = {
        "checkpoint-100/adapter_model.safetensors": b"toy-weights-not-real",
        "checkpoint-100/adapter_config.json": json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}).encode(),
        "checkpoint-100/optimizer.pt": b"toy-optimizer",
        "checkpoint-100/scheduler.pt": b"toy-scheduler",
        "checkpoint-100/rng_state.pth": b"toy-rng",
        "checkpoint-100/trainer_state.json": json.dumps(state).encode(),
        "run_config.json": json.dumps(config).encode(),
    }
    report = {
        "protocol": "qwen9b-v8-stage70-to100-original100-scheduler-v1",
        "stage70_source_run": 44861054303,
        "stage70_artifact": 14693522324,
        "original_training_code_sha": "3eac23cca117a41855f4998399f79a5d9d5f4369",
        "model": "Qwen/Qwen3.5-9B",
        "train_sha256": "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7",
        "trained_step_start": 70,
        "trained_step_end": 100,
        "scheduler_max_steps": 100,
        "failed_checks": [],
        "compute_dtype": "torch.float16",
        "full_v8_completed": True,
        "gaia_evaluated": False,
    }
    for file, key in (
        ("adapter_model.safetensors", "adapter_sha256"),
        ("optimizer.pt", "optimizer_sha256"),
        ("scheduler.pt", "scheduler_sha256"),
        ("rng_state.pth", "rng_sha256"),
    ):
        report[key] = hashlib.sha256(raw["checkpoint-100/" + file]).hexdigest()
    report.update(overrides or {})
    raw["stage100-verified-report.json"] = json.dumps(report).encode()
    for item in missing or []:
        raw.pop(item, None)
    for item, payload in (damaged or {}).items():
        raw[item] = payload
    path = tmp_path / "v8.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in raw.items():
            archive.writestr(name, contents)
    return path


def test_accepted_archive_extracts_only_verified_adapter(tmp_path):
    path = _archive(tmp_path)
    report = validate_stage100_archive(path)
    extracted = extract_verified_adapter(path, tmp_path / "adapter")
    assert report["full_v8_completed"] is True
    assert extracted["adapter_sha256"] == report["adapter_sha256"]
    assert (tmp_path / "adapter" / "adapter_model.safetensors").read_bytes() == b"toy-weights-not-real"
    assert (tmp_path / "adapter" / "adapter_config.json").is_file()
    assert not (tmp_path / "adapter" / "optimizer.pt").exists()


@pytest.mark.parametrize("overrides", [
    {"full_v8_completed": False},
    {"failed_checks": ["compute_dtype"]},
    {"trained_step_end": 90},
    {"original_training_code_sha": "wrong"},
    {"gaia_evaluated": True},
    {"adapter_sha256": "tampered"},
])
def test_invalid_or_unverified_checkpoint_never_starts_gaia(tmp_path, overrides):
    with pytest.raises(ValueError):
        validate_stage100_archive(_archive(tmp_path, overrides=overrides))


@pytest.mark.parametrize("path", ["checkpoint-100/optimizer.pt", "checkpoint-100/rng_state.pth",
                                 "checkpoint-100/scheduler.pt", "checkpoint-100/adapter_config.json"])
def test_missing_full_state_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        validate_stage100_archive(_archive(tmp_path, missing=[path]))


def test_physical_adapter_hash_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError):
        validate_stage100_archive(_archive(tmp_path, damaged={"checkpoint-100/adapter_model.safetensors": b"tampered"}))


def test_untrusted_zip_entries_rejected(tmp_path):
    path = _archive(tmp_path)
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("../escape", b"illegal")
    with pytest.raises(ValueError, match="unsafe"):
        validate_stage100_archive(path)


def test_only_original_bf16_quarantine_can_be_audited_for_evaluation(tmp_path):
    from zipfile import ZipFile
    path = _archive(tmp_path, overrides={
        "full_v8_completed": False,
        "failed_checks": ["compute_dtype"],
        "compute_dtype": "torch.bfloat16",
    })
    # Preserve the old report and alter only synthetic config, as real training did.
    with ZipFile(path, "r") as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    config = json.loads(files["run_config.json"])
    config["compute_dtype"] = "torch.bfloat16"
    files["run_config.json"] = json.dumps(config).encode()
    audit = tmp_path / "qlora-v8-stage100-UNVERIFIED.zip"
    with ZipFile(audit, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    proof = validate_stage100_archive(audit)
    assert proof["evaluation_acceptance"] == "audited_bf16_continuity"
    assert proof["full_v8_completed"] is False  # no rewriting failed Colab report
    assert (tmp_path / "adapter" / "optimizer.pt").exists() is False
    extract_verified_adapter(audit, tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
    # Do not accept arbitrary bf16 artifacts or corrupted optimizer.
    with pytest.raises(ValueError):
        validate_stage100_archive(path)
    files["checkpoint-100/optimizer.pt"] = b"corrupt"
    with ZipFile(audit, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match="SHA mismatch"):
        validate_stage100_archive(audit)
