"""No AppWorld/GPU needed: fixed original 57-pair test contract and stage gates."""
import pytest
from hashlib import sha256

from scripts import eval_appworld_dev57_public_supervisor_get as x


def fake_pairs():
    return [(
      {"official_state_success":i<14,
       "scenario_sha256":sha256(str(i//3).encode()).hexdigest()},
      {"official_state_success":i<19,
       "scenario_sha256":sha256(str(i//3).encode()).hexdigest()}
    ) for i in range(57)]


def test_57_task_19_scene_bootstrap_deterministic_and_positive_when_no_regressions():
    a=fake_pairs()
    one=x.paired_ci(a)
    again=x.paired_ci(a)
    cluster=x.paired_ci(a,cluster=True)
    assert one==again and one["delta"]==5/57
    assert cluster["n_scenarios"]==19 and cluster["unit"]=="scenario_cluster"
    assert one["n_tasks"]==57
    assert x.exact_mcnemar(5,0)==0.0625
    assert x.exact_mcnemar(0,0)==1.0


def test_refuse_partial_and_mismatched_or_duplicate_source_pairs():
    a=fake_pairs()
    with pytest.raises(ValueError):
        x.paired_ci(a[:-1])
    a[1][1]["scenario_sha256"]="mismatch"
    with pytest.raises(ValueError):
        x.paired_ci(a,cluster=True)


def test_source_does_not_redefine_benchmark_or_base_model():
    assert x.SOURCE_SHA==x.base.freeze.__globals__["freeze_official_dev"].__globals__["OFFICIAL_DEV_IDS_SHA"]
    assert x.N==57 and x.SCENARIOS==19
    assert x.MODEL_GGUF_SHA==x.base.MODEL_GGUF_SHA
    assert x.TREATMENT_POLICY!=x.base.STUDENT_POLICY
    assert x.MIN_EFFECT==.05
    assert x.BOOTSTRAP_ROUNDS>=4000


def test_same_original_prompt_contract_is_kept():
    import inspect
    source=inspect.getsource(x.one_task)
    old=inspect.getsource(x.base.task)
    # Same exact user prompt and original fixed model, no new prompt.
    assert '"USER TASK:\\n"+world.task.instruction' in source
    assert '"USER TASK:\\n"+world.task.instruction' in old
    assert 'model="qwen3.5-9b"' in source
    assert 'allow_supervisor_public_reads=True' in source
