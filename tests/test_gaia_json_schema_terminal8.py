from __future__ import annotations

import json
import pytest

from gaia_small_agent.agent.types import ModelRuntimeError
from scripts.gaia_json_schema_terminal8 import (
    FINAL_SCHEMA,ConstrainedTerminalModel,strictly_decode_answer,
    FROZEN_IDS,FROZEN_SHA,aggregate,
)


class FakeResponse:
    def __init__(self,content):
        self.content=content
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices":[{"message":{"content":self.content,"tool_calls":[]}}]}


def test_strict_answer_json_parser_refuses_freeform_fallback():
    assert strictly_decode_answer('{"answer":"Tokyo"}')=="Tokyo"
    assert strictly_decode_answer('{"answer":" 42 "}')=="42"
    for bad in ("Tokyo","{}",'{"answer":null}','{"answer":""}',
                '{"answer":"A","other":"B"}','["A"]'):
        with pytest.raises(ModelRuntimeError):
            strictly_decode_answer(bad)


def test_only_final_request_has_json_schema_and_no_tools(monkeypatch):
    sent=[]
    def fake_post(url,*,json,timeout):
        sent.append((url,json,timeout))
        return FakeResponse('{"answer":"Helsinki"}')
    monkeypatch.setattr("requests.post",fake_post)
    model=ConstrainedTerminalModel()
    turn=model.complete_final(
        [{"role":"system","content":"Evidence first"},
         {"role":"user","content":"Capital of Finland?"}],tools=[{"type":"function","function":{"name":"search"}}]
    )
    assert turn.content=="Helsinki" and not turn.tool_calls
    url,body,timeout=sent[0]
    assert body["response_format"]==FINAL_SCHEMA
    assert body["tools"]==[{"type":"function","function":{"name":"search"}}]
    assert body["tool_choice"]=="none"
    assert body["temperature"]==0 and body["model"]=="qwen3.5-9b"
    assert len(body["messages"])==2
    assert url.endswith("/v1/chat/completions")


def test_prior_gaia_id_freeze_is_independent_of_new_outputs():
    assert len(FROZEN_IDS)==len(set(FROZEN_IDS))==8
    assert len(FROZEN_SHA)==64


def test_aggregate_fails_closed_for_partial_repeated_cases(tmp_path):
    p=tmp_path/"missing.json";p.write_text("{}")
    with pytest.raises(ValueError,match="four physical"):
        aggregate([p])
