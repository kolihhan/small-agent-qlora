"""Frozen aggregate-only coverage association: synthetic cases without task content."""
import json
import pytest
from scripts.diagnose_gaia100_read_outcomes_posthoc import CATEGORIES,bucket,summarize


def _call(i,source):
    return {"kind":"tool_call","data":{"id":str(i),"name":"read","arguments":{"source":source}}}

def _result(i,ok):
    return {"kind":"tool_result","data":{"id":str(i),"name":"read","ok":ok,"error_code":None if ok else "READ_ERROR",
                                        "content":"untrusted actual text which must not appear in summary"}}

def _row(events,correct):
    return {"trace":events,"correct":correct,"completed":True,"stop_reason":"final",
            "answer":"private string","expected_answer":"hidden gold"}

def test_predefined_categories_and_no_task_export():
    cases=[
        _row([],True),
        _row([_call(1,"a"),_result(1,False)],False),
        _row([_call(1,"a"),_result(1,True)],True),
        _row([_call(1,"a"),_result(1,True),_call(2,"b"),_result(2,True)],False),
    ]
    assert [bucket(x["trace"]) for x in cases]==list(CATEGORIES)
    out=summarize(cases)
    assert out["n"]==4 and out["correct"]==2
    assert all(out["categories"][c]["n"]==1 for c in CATEGORIES)
    assert "private string" not in json.dumps(out)
    assert "hidden gold" not in json.dumps(out)


def test_repeat_read_same_source_does_not_fake_multihop():
    trace=[_call(1,"a"),_result(1,True),_call(2,"a"),_result(2,True)]
    assert bucket(trace)=="one_distinct_successful_read_source"


def test_correctness_type_required():
    with pytest.raises(ValueError):
        summarize([_row([],1)])
