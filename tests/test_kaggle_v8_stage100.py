"""Kaggle T4 stage70 continuation must preserve the verified original run contract.

CPU-only checks: these do not download Qwen weights or use Kaggle quota.
"""
from __future__ import annotations

from pathlib import Path
import ast

from scripts import kaggle_v8_stage100_remote as k


def test_v8_kaggle_single_t4_original_state_hashes_and_training_source():
    assert k.MODEL == "Qwen/Qwen3.5-9B"
    assert k.STAGE70_RUN == 44861054303
    assert k.STAGE70_ARTIFACT == 14693522324
    assert k.SOURCE_SHA == "3eac23cca117a41855f4998399f79a5d9d5f4369"
    assert k.TRAIN_SHA == "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
    assert k.STAGE70_WEIGHTS_SHA == "214ca477a95c8dd9c4395d1093ada147eb7fed62d33b3781f846d28d54f4c8c8"
    assert k.STAGE70_OPTIM_SHA == "d5e928f8029d99f4dd94e805a65a7b33973dfe04ed407da9a81154053ec6ae99"
    assert k.STAGE70_SCHED_SHA == "6022b1ed992ffdf30d48cb9476b12819081d7b2f69813a50222fc783ae361512"
    assert k.STAGE70_RNG_SHA == "aa78619ac2c20eff4b2694524a8b7962d5b5d6cb7b21edd426f38a2a0ec54c19"


def test_kaggle_private_input_and_saved_output_never_enter_training_source():
    assert k.DATASET_ID == "kolihhan/qlora-v8-stage70-verified-20261121"
    # Windows runners render POSIX paths with backslashes; normalize only for assertions.
    slash = lambda p: str(p).replace(chr(92), "/")
    assert slash(k.BUNDLE).startswith("/kaggle/input/")
    assert slash(k.TRAIN).startswith("/kaggle/input/")
    assert slash(k.OUT).startswith("/kaggle/temp/")
    assert slash(k.ARCHIVE).startswith("/kaggle/working/")
    assert not k.ARCHIVE.exists()


def test_training_args_match_frozen_v8_original_resume_and_single_t4():
    src = Path(k.__file__).read_text()
    ast.parse(src)
    assert "torch.cuda.device_count()!=1" in src
    assert "torch.cuda.get_device_capability(0)!=(7,5)" in src
    assert '"T4" not in torch.cuda.get_device_name(0).upper()' in src
    assert 'max_length=3200,epochs=1.0,learning_rate=2e-5' in src
    assert 'lora_r=16,lora_alpha=32,lora_dropout=0.05' in src
    assert 'max_steps=MAX_STEPS,save_steps=10,gradient_accumulation_steps=8' in src
    assert 'lora_target_modules="all-linear",exposure_count=None' in src
    assert 'require_full_prompt=True,preserve_large_half_parameters=True' in src
    assert 'pause_at_step=None,resume_from_checkpoint=OUT/"checkpoint-70"' in src


def test_missing_private_kaggle_dataset_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(k,"TRAIN",tmp_path/"missing-train.jsonl")
    monkeypatch.setattr(k,"AUDIT",tmp_path/"missing-audit.json")
    monkeypatch.setattr(k,"BUNDLE",tmp_path/"missing-checkpoint.zip")
    monkeypatch.setattr(k,"SOURCE_MANIFEST",tmp_path/"missing-manifest.json")
    monkeypatch.setattr(k,"OUT",tmp_path/"checkpoint")
    monkeypatch.setattr(k,"ARCHIVE",tmp_path/"out.zip")
    try:
        k.verify_and_restore()
    except FileNotFoundError as err:
        assert "private Kaggle stage70 dataset" in str(err)
    else:
        raise AssertionError("missing checkpoint must not proceed")


def _synthetic_stage70(tmp_path, monkeypatch, *, zipped=False):
    """Build isolated toy bytes; do not download the real private model."""
    import hashlib
    import json
    import zipfile
    root = tmp_path / "kaggle-input"
    root.mkdir()
    monkeypatch.setattr(k, "DATASET_ROOT", root)
    monkeypatch.setattr(k, "TRAIN", root / "qlora-v8-train.jsonl")
    monkeypatch.setattr(k, "AUDIT", root / "qlora-v8-audit.json")
    monkeypatch.setattr(k, "SOURCE_MANIFEST", root / "stage70-kaggle-transport-manifest.json")
    monkeypatch.setattr(k, "BUNDLE", root / "qlora-v8-stage70-checkpoint.zip")
    monkeypatch.setattr(k, "OUT", tmp_path / "work" / "qlora-v8-adapter")
    monkeypatch.setattr(k, "ARCHIVE", tmp_path / "work" / "unused-output.zip")
    k.TRAIN.write_bytes(b'{"toy":"training"}\\n')
    monkeypatch.setattr(k, "TRAIN_SHA", hashlib.sha256(k.TRAIN.read_bytes()).hexdigest())
    audit = {
        "train_sha256": k.TRAIN_SHA, "train_actions": 798,
        "train_parents": 304, "validation_parents": 35,
        "protected_question_hashes_loaded": 165,
        "training_ready_no_gpu_yet": True, "no_gaia_training": True,
    }
    k.AUDIT.write_text(json.dumps(audit))
    proof = {
        "protocol": "qwen9b-v8-continued-step35-to70-original100-scheduler-v1",
        "model": k.MODEL,
        "original_training_code_sha": k.SOURCE_SHA,
        "train_sha256": k.TRAIN_SHA,
        "original_stage35_run": 44781908112,
        "trained_step_start": 35, "trained_step_end": 70,
        "scheduler_max_steps": 100,
        "full_v8_completed": False, "gaia_evaluated": False,
    }
    config = {"max_steps": 100, "pause_at_step": 70, "max_length": 3200,
              "training_examples": 798}
    state = {"global_step": 70, "max_steps": 100}
    data = {
        "stage70-verified-report.json": json.dumps(proof).encode(),
        "run_config.json": json.dumps(config).encode(),
        "checkpoint-70/trainer_state.json": json.dumps(state).encode(),
        "checkpoint-70/adapter_config.json": b'{"r": 16}',
    }
    for name, key in (
        ("adapter_model.safetensors", "STAGE70_WEIGHTS_SHA"),
        ("optimizer.pt", "STAGE70_OPTIM_SHA"),
        ("scheduler.pt", "STAGE70_SCHED_SHA"),
        ("rng_state.pth", "STAGE70_RNG_SHA"),
    ):
        data["checkpoint-70/" + name] = ("toy-" + name).encode()
        monkeypatch.setattr(k, key, hashlib.sha256(data["checkpoint-70/" + name]).hexdigest())
    if zipped:
        with zipfile.ZipFile(k.BUNDLE, "w") as z:
            for name, raw in data.items():
                z.writestr(name, raw)
        zip_bytes = k.BUNDLE.stat().st_size
        zip_sha = k.sha_file(k.BUNDLE)
        monkeypatch.setattr(k, "STAGE70_FILE_SIZE", zip_bytes)
    else:
        # Test an extra nesting level as Kaggle's ZIP auto-extraction may use one.
        prefix = root / "qlora-v8-stage70-checkpoint"
        for name, raw in data.items():
            dest = prefix / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        zip_bytes = k.STAGE70_FILE_SIZE
        zip_sha = "b" * 64
    manifest = {
        "protocol": "private-kaggle-original-qlora-v8-step70-v1",
        "stage70_artifact": k.STAGE70_ARTIFACT,
        "stage70_source_run": k.STAGE70_RUN,
        "dataset_id": k.DATASET_ID,
        "stage70_zip_bytes": zip_bytes,
        "stage70_zip_sha256": zip_sha,
    }
    k.SOURCE_MANIFEST.write_text(json.dumps(manifest))
    return data


def test_kaggle_auto_unpacked_full_optimizer_restores_with_hashes(tmp_path, monkeypatch):
    data = _synthetic_stage70(tmp_path, monkeypatch)
    proof = k.verify_and_restore()
    assert proof["trained_step_end"] == 70
    for name, raw in data.items():
        if name.startswith("checkpoint-70/"):
            assert (k.OUT / name).read_bytes() == raw


def test_kaggle_original_zip_backward_compatible(tmp_path, monkeypatch):
    data = _synthetic_stage70(tmp_path, monkeypatch, zipped=True)
    k.verify_and_restore()
    for name, raw in data.items():
        if name.startswith("checkpoint-70/"):
            assert (k.OUT / name).read_bytes() == raw


def test_kaggle_unpacked_corrupt_optimizer_fails_closed(tmp_path, monkeypatch):
    import pytest
    _synthetic_stage70(tmp_path, monkeypatch)
    opt = next(k.DATASET_ROOT.rglob("checkpoint-70/optimizer.pt"))
    opt.write_bytes(b"corrupted-state")
    with pytest.raises(ValueError, match="tensor state SHA mismatch"):
        k.verify_and_restore()


def test_kaggle_unpacked_ambiguous_checkpoints_fails_closed(tmp_path, monkeypatch):
    import pytest
    _synthetic_stage70(tmp_path, monkeypatch)
    other = k.DATASET_ROOT / "second" / "checkpoint-70"
    other.mkdir(parents=True)
    (other / "trainer_state.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="ambiguous"):
        k.verify_and_restore()
