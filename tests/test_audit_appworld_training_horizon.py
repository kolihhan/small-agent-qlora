"""No AppWorld dependency required: source-count privacy/quality contracts."""
import pytest
from scripts import audit_appworld_training_horizon as a


def _calls(n,app="spotify"):
    return [{"url":f"/{app}/api_{i}","method":"GET"} for i in range(n)]

def _rows(prefix,n,depth):
    return [(f"{prefix}{i}_1",_calls(depth,"spotify" if i%2 else "venmo"))
            for i in range(n)]

def test_business_api_depth_not_inflated_by_supervisor_or_admin_calls():
    rows=_calls(6)+[
        {"url":"/admin/setup","method":"POST"},
        {"url":"/supervisor/complete_task","method":"POST"},
        {"url":"/api_docs/show","method":"GET"},
    ]
    result=a._call_stats(rows)
    assert result=={"all_calls":9,"business_calls":6,
                    "business_apps":1,"malformed_calls":0}

def test_real_source_feasibility_reports_counts_only_and_separates_scenarios():
    t=_rows("train",60,8)
    d=_rows("dev",25,2)
    result=a.audit_splits(t,d)
    assert result["train"]["tasks_with_6plus_business_calls"]==60
    assert result["train"]["tasks_with_8plus_business_calls"]==60
    assert result["dev"]["tasks_with_6plus_business_calls"]==0
    assert result["train"]["histogram"]=={"8-11":60}
    assert result["no_model"] and result["no_gpu"] and not result["training_labels_verified"]
    assert "cases" not in result and "task_ids" not in result["train"]
    assert len(result["train"]["task_ids_sha256"])==64

def test_source_overlap_scenario_leakage_and_missing_splits_fail():
    t=_rows("train",60,6)
    d=_rows("dev",25,6)
    with pytest.raises(ValueError):
        a.audit_splits(t,t)
    with pytest.raises(ValueError):
        a.audit_splits(t,d[:1])
    d[0]=("train0_2",_calls(7))
    with pytest.raises(ValueError):
        a.audit_splits(t,d)

def test_invalid_gold_reference_entries_not_converted_to_label():
    with pytest.raises(ValueError):
        a._call_stats(None)
    counts=a._call_stats([{"url":"malformed"},123,{"url":"/gmail/send","method":"POST"}])
    assert counts["business_calls"]==1 and counts["malformed_calls"]==2

def test_no_task_id_duplicates_even_without_split_overlap():
    with pytest.raises(ValueError):
        a.summarize("train",[("repeat_1",[]),("repeat_1",[])])
