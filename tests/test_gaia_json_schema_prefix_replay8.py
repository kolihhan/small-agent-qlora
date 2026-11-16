from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gaia_json_schema_prefix_replay8 import (
    OriginalFinalCapturer, check_final_result, aggregate, FROZEN_IDS,
    FROZEN_SHA, PROTOCOL,
)


def test_original_final_capture_copies_before_backend_call(monkeypatch):
    observed={}
    def fake_complete(self,messages,tools):
        observed["seen"]=messages[0]["content"]
        return "original_fallback"
    monkeypatch.setattr("gaia_small_agent.model.openai_compat.OpenAICompatModel.complete_final",fake_complete)
    model=OriginalFinalCapturer()
    messages=[{"role":"system","content":"use given evidence"},{"role":"user","content":"who?"}]
    tools=[{"type":"function","function":{"name":"read"}}]
    assert model.complete_final(messages,tools)=="original_fallback"
    messages[0]["content"]="changed after capture"
    assert model.original_final_input[0][0]["content"]=="use given evidence"
    assert model.original_final_input[1]==tools


def test_constrained_string_still_requires_valid_final_not_a_tool_call():
    assert check_final_result("Helsinki")==(True,"final")
    assert check_final_result("<tool_call>search</tool_call>")==(False,"invalid_final")
    assert check_final_result("   ")==(False,"empty_final")


def test_full_8_gate_rejects_empty_or_wrong_shards(tmp_path):
    with pytest.raises(ValueError,match="four physical"):
        aggregate([])
    p=tmp_path/"notjson.json"
    p.write_text(json.dumps({"shard":0}))
    with pytest.raises(ValueError,match="four physical"):
        aggregate([p])


def test_source_fixed_before_any_terminal_replay():
    assert len(FROZEN_IDS)==len(set(FROZEN_IDS))==8
    assert len(FROZEN_SHA)==64
    assert "EXACT-PREFIX" in PROTOCOL.upper().replace("_","-") or "FIXED-PREFIX" in PROTOCOL.upper()
