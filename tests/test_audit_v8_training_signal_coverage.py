"""Synthetic-only contract tests for the v8 SFT signal coverage census."""
import json

import pytest

from scripts.audit_v8_training_signal_coverage import audit_rows


def _row(ident, *, error=False):
    observations = [
        {"role":"system","content":"Static instruction"},
        {"role":"user","content":"Fictional task "+ident},
        {"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":{"source":"fake.txt"}}}]},
        {"role":"tool","tool_call_id":"a","content":"ERROR[SEARCH_FIXTURE_MISS]: fake source unavailable" if error else "Fictional evidence number 4"},
        {"role":"assistant","tool_calls":[{"id":"b","type":"function","function":{"name":"read","arguments":{"source":"fake2.txt"}}}]},
        {"role":"tool","tool_call_id":"b","content":"Fictional evidence number 9"},
        {"role":"assistant","content":"Fictional answer"},
    ]
    return {
        "task_id":ident,"source":"finqa-train","verified":True,
        "messages":observations,
        "training_action_indices":[2,4,6],
    }


def test_aggregation_probes_recovery_and_multitool(tmp_path):
    rows=[_row("alpha",error=True),_row("beta",error=False)]
    path=tmp_path/"source.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    summary,ids,hashes=audit_rows(path,expected_parents=2,expected_actions=6)
    group=summary["per_source"]["finqa_observed"]
    assert group["selected_actions"]==6
    assert group["selected_tool_action_directly_after_tool_error"]==1
    assert group["selected_final_after_2plus_observations"]==2
    assert group["selected_tool_after_nonerror_observation"]==1
    assert "Fictional" not in json.dumps(summary)
    assert len(ids)==len(hashes)==2


def test_refuses_overlapping_parent_or_question(tmp_path):
    path=tmp_path/"source.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [_row("alpha"),_row("alpha")]))
    with pytest.raises(ValueError,match="duplicate task"):
        audit_rows(path,expected_parents=2,expected_actions=6)


def test_rejects_unverified_label_and_missing_indices(tmp_path):
    path=tmp_path/"source.jsonl"
    row=_row("alpha")
    row["verified"]=False
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match="Unverified"):
        audit_rows(path,expected_parents=1,expected_actions=3)
    row["verified"]=True
    row["training_action_indices"]=[2,4,4]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match="Invalid original"):
        audit_rows(path,expected_parents=1,expected_actions=3)
