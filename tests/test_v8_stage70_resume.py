"""CPU-only contract: v8 resumes original checkpoint, never swaps models or sources."""
from scripts import colab_v8_stage70_remote as r


def test_training_schedule_and_sha_are_original():
    assert r.MODEL == "Qwen/Qwen3.5-9B"
    assert (r.PAUSE,r.MAX_STEPS) == (70,100)
    assert r.SOURCE_SHA == "3eac23cca117a41855f4998399f79a5d9d5f4369"
    assert r.STAGE1_WEIGHTS_SHA == "3fe28aeaefe48d585c229ec56b0637542158fbdc98a29b09d8539ecee49e58c5"
    assert r.STAGE1_TRAIN_SHA == "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"


def test_stage70_runner_has_checkpoint_resume_and_no_gaia_training():
    from inspect import getsource
    code=getsource(r.main)
    assert 'resume_from_checkpoint=OUT/"checkpoint-35"' in code
    assert 'pause_at_step=PAUSE' in code
    assert 'require_full_prompt=True' in code
    assert 'preserve_large_half_parameters=True' in code
    assert 'audit.get("no_gaia_training") is not True' in code
    assert 'lora_target_modules="all-linear"' in code
