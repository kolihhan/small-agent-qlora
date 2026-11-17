"""Check original request method/data dispatch without loading AppWorld task code."""
from types import SimpleNamespace
import pytest
from scripts import replay_appworld_train_api_trace as m

class FakeRequester:
    def __init__(self):self.seen=[]
    def get(self,**kwargs):self.seen.append(("get",kwargs));return {"ok":True}
    def post(self,**kwargs):self.seen.append(("post",kwargs));return {"ok":True}
    def patch(self,**kwargs):self.seen.append(("patch",kwargs));return {"ok":True}

def test_replay_records_actual_original_request_methods_and_data_privately():
    req=FakeRequester();world=SimpleNamespace(requester=req)
    calls=[{"method":"get","url":"/spotify/list","data":{"offset":0}},
           {"method":"post","url":"/venmo/send","data":{"token":"fictional"}}]
    assert m.replay_calls(world,calls)==2
    assert [x[0] for x in req.seen]==["get","post"]
    assert req.seen[1][1]["show"] is False
    assert req.seen[1][1]["raise_on_failure"] is False

def test_unsupported_untrusted_method_or_absent_original_source_fails_closed():
    obj=SimpleNamespace(requester=FakeRequester())
    for bad in (None,[],[{"method":"rm","url":"/x","data":{}}],
                [{"method":"post","url":"","data":{}}],
                [{"method":"get","url":"/x","data":"bad"}],
                [{}]):
        with pytest.raises(ValueError):
            m.replay_calls(obj,bad)

def test_no_test_or_dev_or_model_training():
    assert m.PROTOCOL.endswith("STATE-REPLAY-V1")
    assert m.MAX_REQUESTS==400
    assert m.REF_PROTOCOL.endswith("PILOT-V1")
