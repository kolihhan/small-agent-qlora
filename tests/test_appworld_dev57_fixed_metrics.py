"""CPU-only tests: permanent AppWorld DEV57 roster, staged metrics, no K inflation."""
from hashlib import sha256
from types import SimpleNamespace
import pytest

from scripts import appworld_dev57_fixed_metrics as m


def test_main_roster_fails_closed_on_unverified_task_ids():
    ids=[f"scene{i//3}_{i%3+1}" for i in range(57)]
    assert len({x.split("_",1)[0] for x in ids})==19
    with pytest.raises(ValueError,match="source"):
        m.freeze_official_dev(ids)
    assert m.N==57 and m.SCENARIOS==19
    assert m.EXPOSED_PILOTS==8


def test_one_record_counts_stages_without_exporting_trace():
    def event(kind,data):
        return SimpleNamespace(kind=kind,data=data)
    trace=[
      event("tool_call",{"name":"appworld_call","arguments":{"token":"SECRET"}}),
      event("tool_result",{"name":"appworld_call","ok":False,
                           "error_code":"HTTP_ERROR","content":"SENSITIVE"}),
      event("tool_call",{"name":"appworld_docs","arguments":{"app":"venmo"}}),
      event("tool_result",{"name":"appworld_docs","ok":True,"error_code":None}),
      event("tool_call",{"name":"appworld_call","arguments":{"token":"SECRET2"}}),
      event("tool_result",{"name":"appworld_call","ok":True,"error_code":None}),
      event("final",{"answer":"PASSWORD SHOULD NOT BE LOGGED"})
    ]
    result=SimpleNamespace(
        trace=trace,stop_reason="final",completed=True,
        metrics=SimpleNamespace(steps=4,tool_calls=3,tool_successes=2,
                                tool_errors=1,invalid_tool_calls=0))
    row=m.extract_stage_record("alpha_1",result,api_calls=2,
          real_http_failures=1,official_state_success=False,
          state_tests_pass=2,state_tests_total=5,
          observed_reference_call_count=9)
    assert row["http_error_observations"]==1
    assert row["http_error_followed_by_successful_api"]==1
    assert row["reference_api_depth_bucket_for_postrun_only"]=="6-11"
    assert row["tool_call_format_valid"]==3
    assert row["stop_reason"]=="final"
    assert row["raw_trace_or_task_text_exported"] is False
    assert "SECRET" not in str(row) and "PASSWORD" not in str(row)


def fake_cohort():
    tasks=[f"scene{i//3}_{i%3+1}" for i in range(57)]
    hashes=[sha256(t.encode()).hexdigest() for t in tasks]
    meta={
        "protocol":m.PROTOCOL,"n":57,"model_sha256":m.MODEL_GGUF_SHA,
        "task_ids_sha256":m.OFFICIAL_DEV_IDS_SHA,
        "task_hashes_sha256":sha256("\n".join(sorted(hashes)).encode()).hexdigest(),
        "previously_exposed":8
    }
    rows=[]
    for i,t in enumerate(tasks):
        rows.append({
          "stage_version":m.PROTOCOL,
          "task_sha256":sha256(t.encode()).hexdigest(),
          "scenario_sha256":sha256(t.split("_",1)[0].encode()).hexdigest(),
          "raw_trace_or_task_text_exported":False,
          "official_state_success":i<11,
          "official_tests_pass":i%3,"official_tests_total":4,
          "model_decision_steps":5,"tool_calls":5,
          "tool_call_format_valid":4,"tool_successes":4,"tool_errors":1,
          "api_calls":3,"api_http_failures":1,
          "http_error_observations":1,
          "http_error_followed_by_successful_api":int(i%2==0),
          "final_normal":i<28,"stop_reason":"final" if i<28 else "model_error",
          "reference_api_depth_bucket_for_postrun_only":(
             "0-5" if i<8 else "6-11" if i<20 else "12+")
        })
    return meta,hashes,rows


def test_only_full_unique_fixed_roster_can_aggregate():
    meta,hashes,rows=fake_cohort()
    r=m.aggregate_dev57(rows,meta,expected_task_hashes=hashes)
    assert r["n"]==57 and r["state_success"]==11
    assert r["official_state_test_total"]==228
    assert r["tool_format_validity_rate"]==0.8
    assert r["api_http_failure_rate"]==1/3
    assert r["post_http_recovery_rate"]>0.5
    assert r["previously_exposed_in_main"]==8
    assert r["difficulty_postrun_breakdown"]["0-5"]["n"]==8
    assert not r["no_GAIA_score_claimed"] is False
    assert r["not_statistically_powered_for_5pp"]
    with pytest.raises(ValueError):
        m.aggregate_dev57(rows[:-1],meta,expected_task_hashes=hashes)
    with pytest.raises(ValueError):
        m.aggregate_dev57(rows[:-1]+[rows[0]],meta,expected_task_hashes=hashes)
    with pytest.raises(ValueError):
        m.aggregate_dev57(rows,meta,expected_task_hashes=hashes[:-1])
    rows[0]["raw_trace_or_task_text_exported"]=True
    with pytest.raises(ValueError):
        m.aggregate_dev57(rows,meta,expected_task_hashes=hashes)


def test_fixed_primary_not_moving_hotpot_or_gaia():
    assert m.PROTOCOL.startswith("SMALL-AGENT-APPWORLD-DEV57")
    assert m.N==57 and m.SCENARIOS==19
    assert "tool_call_validity" in m.STAGES
    assert "post_error_recovery" in m.STAGES
    assert "official_state_success" in m.STAGES
