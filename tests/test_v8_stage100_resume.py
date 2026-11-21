"""CPU-only integrity checks before allocating the next Colab GPU."""

from inspect import getsource
from pathlib import Path
from scripts import colab_v8_stage100_remote as r


def test_original_model_source_and_verified_step70_unchanged():
    assert r.MODEL == "Qwen/Qwen3.5-9B"
    assert r.MAX_STEPS == 100
    assert r.SOURCE_SHA == "3eac23cca117a41855f4998399f79a5d9d5f4369"
    assert r.TRAIN_SHA == "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
    assert r.STAGE70_RUN == 44861054303
    assert r.STAGE70_ARTIFACT == 14693522324
    assert r.STAGE70_FILE_SIZE == 539885434
    assert len(r.STAGE70_WEIGHTS_SHA) == 64
    assert len(r.STAGE70_OPTIM_SHA) == 64
    assert len(r.STAGE70_SCHED_SHA) == 64
    assert len(r.STAGE70_RNG_SHA) == 64


def test_final30_resumes_full_optimizer_state_without_training_gaia():
    code=getsource(r.main) + getsource(r.export_stage100_checkpoint)
    assert "restore_verified_step70()" in code
    assert 'resume_from_checkpoint=OUT / "checkpoint-70"' in code
    assert "pause_at_step=None" in code
    assert "save_steps=10" in code
    assert "max_steps=MAX_STEPS" in code
    assert 'require_full_prompt=True' in code
    assert 'preserve_large_half_parameters=True' in code
    assert 'audit.get("no_gaia_training") is not True' in code
    assert "lora_target_modules=\"all-linear\"" in code
    assert '"checkpoint-100"' in code
    acceptance=getsource(r.export_stage100_checkpoint)
    assert '"full_v8_completed": verified' in acceptance
    assert '"gaia_evaluated": False' in acceptance


def test_stage70_restore_checks_physical_optimizer_scheduler_rng():
    code=getsource(r.restore_verified_step70)
    assert "STAGE70_OPTIM_SHA" in code
    assert "STAGE70_SCHED_SHA" in code
    assert "STAGE70_RNG_SHA" in code
    assert "trainer_state.json" in code
    assert 'state.get("global_step") == 70' in code
    assert "z.extract(item, OUT)" in code


def test_workflow_uses_one_original_source_and_stage70_artifact():
    source=Path(".github/workflows/qlora-v8-stage100-colab.yml").read_text()
    assert "run-id: 43619541894" in source
    assert "run-id: 44861054303" in source
    assert "qlora-v8-stage70-full-state-if-trained" in source
    assert "qlora-v8-stage100-full-state-if-trained" in source
    assert "qlora-v8-durable-stage35-with-optimizer-rng-if-trained" not in source
    assert "scripts/colab_v8_stage100_remote.py" in source
    assert 'colab --auth=adc usage' in source
    assert "trap cleanup EXIT" in source
    assert "ORIGINAL_QWEN9B_STAGE100_FULL_TRAINING_VERIFIED" in source


def _toy_checkpoint100(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(r, "OUT", tmp_path / "adapter")
    monkeypatch.setattr(r, "ARCHIVE", tmp_path / "exported" / "verified.zip")
    monkeypatch.setattr(r, "QUARANTINE_ARCHIVE", tmp_path / "exported" / "unverified.zip")
    ckpt = r.OUT / "checkpoint-100"
    ckpt.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json",
                 "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (ckpt / name).write_bytes(("toy-" + name).encode())
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 100, "max_steps": 100}))
    cfg = {
        "model": r.MODEL, "compute_dtype": "torch.float16",
        "max_steps": 100, "pause_at_step": None, "save_steps": 10,
        "training_examples": 798,
        "resume_from_checkpoint": str(r.OUT / "checkpoint-70"),
    }
    (r.OUT / "run_config.json").write_text(json.dumps(cfg))
    return cfg


def test_colab_mismatch_quarantines_complete_optimizer_without_false_success(tmp_path, monkeypatch):
    import json
    import zipfile
    cfg = _toy_checkpoint100(tmp_path, monkeypatch)
    cfg["training_examples"] = 797
    (r.OUT / "run_config.json").write_text(json.dumps(cfg))
    report = r.export_stage100_checkpoint(training_returned=True, duration_s=0)
    assert report["full_v8_completed"] is False
    assert report["failed_checks"] == ["training_examples"]
    assert not r.ARCHIVE.exists()
    assert r.QUARANTINE_ARCHIVE.exists()
    with zipfile.ZipFile(r.QUARANTINE_ARCHIVE) as z:
        for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                     "rng_state.pth", "trainer_state.json"):
            assert "checkpoint-100/" + name in z.namelist()
        assert json.loads(z.read("stage100-verified-report.json"))["full_v8_completed"] is False


def test_colab_success_exports_verified_complete_optimizer(tmp_path, monkeypatch):
    import json
    _toy_checkpoint100(tmp_path, monkeypatch)
    report = r.export_stage100_checkpoint(training_returned=True, duration_s=0)
    assert report["full_v8_completed"] is True
    assert report["trained_step_end"] == 100
    assert report["failed_checks"] == []
    assert r.ARCHIVE.exists()
    assert not r.QUARANTINE_ARCHIVE.exists()


def test_colab_exception_cannot_promote_full_checkpoint(tmp_path, monkeypatch):
    _toy_checkpoint100(tmp_path, monkeypatch)
    report = r.export_stage100_checkpoint(training_returned=False, duration_s=0)
    assert report["full_v8_completed"] is False
    assert "training_returned" in report["failed_checks"]
    assert r.QUARANTINE_ARCHIVE.exists()


def test_workflow_preserves_unverified_archive_when_remote_returns_nonzero():
    source = Path(".github/workflows/qlora-v8-stage100-colab.yml").read_text()
    assert "qlora-v8-stage100-UNVERIFIED.zip" in source
    assert "exported/qlora-v8-stage100-UNVERIFIED.zip" in source
    assert "if-no-files-found: warn" in source
