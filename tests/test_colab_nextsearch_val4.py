"""Dry test of heldout Base vs Adapter GAIA-independent scoring/roster."""
import pytest
from scripts.colab_nextsearch_gpu8_matched_val4 import (
    aggregate,initial_prompt,ADAPTER_SHA,COHORT_SHA,N
)


def val_cases(rescues=(0,2),regressions=(),errors=()):
    rows=[]
    for idx in range(N):
        old=(idx==1)
        new=(old or idx in rescues) and idx not in regressions
        arms={}
        for name,exact in (("original",old),("trained8",new)):
            arms[name]={
                "strict_exact":exact,"tool_calls":2,"tool_errors":0,
                "stop_reason":"model_error" if (idx,name) in errors else "final",
            }
        rows.append({"id":str(idx),"source":"asearcher" if idx%2==0 else "taskcraft",
                     "arms":arms})
    return rows


def test_uses_same_original_prompt_and_no_reference_leakage():
    text=initial_prompt("A real user question?")
    assert text.endswith("Question: A real user question?")
    assert "gold" not in text.casefold()
    assert "source" in text.casefold()


def test_two_rescues_two_sources_original_adapter_no_regression_screen():
    x=aggregate(val_cases(rescues=(0,3)))
    assert x["source_cohort_sha"]==COHORT_SHA
    assert x["adapter_sha"]==ADAPTER_SHA
    assert x["cases"]==N
    assert x["base_strict_exact"]==1 and x["adapter_strict_exact"]==3
    assert x["pass_small_screen_only"]
    assert x["promoted"] is False
    assert x["gaia100_tested"] is False


def test_recovery_must_not_hide_regression_or_model_error():
    assert not aggregate(val_cases(rescues=(0,2),regressions=(1,)))["pass_small_screen_only"]
    assert not aggregate(val_cases(rescues=(0,3),errors=((3,"trained8"),)))["pass_small_screen_only"]


def test_incomplete_real_pairs_fail_closed():
    rows=val_cases(rescues=(0,3))[:3]
    with pytest.raises(ValueError,match="missing heldout"):
        aggregate(rows)
