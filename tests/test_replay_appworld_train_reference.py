"""CPU only: test privacy, deterministic selection, source gate (no AppWorld installed)."""
from types import SimpleNamespace
import pytest
from scripts import replay_appworld_train_reference as r

def _sample(idx,callcount=14,twos=True):
    calls=[{"url":"/spotify/read"} for _ in range(callcount)]
    if twos:calls[0]={"url":"/venmo/read"}
    return (f"scenario{idx}_1",SimpleNamespace(api_calls=calls))

def test_selects_eight_independent_two_app_train_scenarios():
    chosen=r.choose_train_entries([_sample(i) for i in range(12)])
    assert len(chosen)==8 and len(set(x.split("_")[0] for x in chosen))==8
    assert all(x.startswith("scenario") for x in chosen)
    again=r.choose_train_entries([_sample(i) for i in reversed(range(12))])
    assert chosen==again

def test_ignores_shallow_or_single_app_or_same_scenario():
    entries=[_sample(i,callcount=3) for i in range(6)]
    entries += [_sample(i+10,twos=False) for i in range(8)]
    entries += [_sample(99),("scenario99_2",SimpleNamespace(api_calls=_sample(99)[1].api_calls))]
    with pytest.raises(ValueError):
        r.choose_train_entries(entries)

def test_private_source_record_is_not_in_export_contract():
    assert r.N==8 and r.MIN_CALLS==12
    assert r.PROTOCOL.endswith("PILOT-V1")
    assert not hasattr(r,"EVALUATION_TASK_TEXT")
