"""Only anonymized real-task failure categories; CI uses fake trace events."""
from gaia_small_agent.agent.types import TraceEvent
from scripts.appworld_train_observed_failure_status_budget import first_failure,trajectory


def test_first_observed_failure_is_not_implicitly_causal():
    t=[
       TraceEvent("tool_result",1,{"name":"appworld_docs","ok":True,"error_code":None}),
       TraceEvent("tool_result",2,{"name":"appworld_call","ok":False,"error_code":"HTTP_ERROR",
                                    "content":"PRIVATE_SECRET"}),
       TraceEvent("tool_result",3,{"name":"appworld_call","ok":True,"error_code":None}),
       TraceEvent("tool_result",4,{"name":"appworld_call","ok":False,"error_code":"HTTP_ERROR"}),
    ]
    events=[{"model_step":2,"api_call_index":1,"http_status":401},
            {"model_step":4,"api_call_index":3,"http_status":422}]
    x=trajectory(t,events,20)
    assert x["first_observed_tool_failure"]=={
        "step":2,"tool":"appworld_call","code":"HTTP_ERROR"}
    assert x["http_error_counts"]=={"401":1,"422":1}
    assert x["any_successful_API_after_first_error"]
    assert x["first_error_repeated_afterwards"]==1
    assert not x["causal_first_blocker_verified"]
    assert "PRIVATE_SECRET" not in repr(x)


def test_no_tool_error_not_same_as_success():
    tr=[TraceEvent("tool_result",2,{"name":"appworld_docs","ok":True})]
    x=trajectory(tr,[],20)
    assert first_failure(tr) is None
    assert x["first_observed_tool_failure"] is None
    assert x["http_error_counts"]=={}
    assert not x["causal_first_blocker_verified"]


def test_error_code_allowlist_prevents_raw_export():
    tr=[TraceEvent("tool_result",1,{"name":"secret_private_tool",
                                   "ok":False,"error_code":"UNKNOWN_PRIVATE_123"})]
    x=trajectory(tr,[],20)
    assert x["first_observed_tool_failure"]=={
        "step":1,"tool":"other","code":"OTHER_TOOL_ERROR"}
    assert "UNKNOWN_PRIVATE" not in repr(x)
