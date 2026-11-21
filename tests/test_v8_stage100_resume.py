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
    code=getsource(r.main)
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
    assert '"full_v8_completed": True' in code
    assert '"gaia_evaluated": False' in code


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
