from hashlib import sha256
import json

from gaia_small_agent.agent.types import TraceEvent
from scripts import eval_2wiki_serializer_confirm48 as source48
from scripts.eval_2wiki16_v7_evidence_prompt_ab import (
    BASE_PROMPT, TREAT_PROMPT, MODEL, BASE_SHA, V7_SHA,
    h, read_observations, GATE, N, SHARDS
)


def test_original_qwen9b_v7_and_agent_budget_frozen():
    assert MODEL == "qwen3.5-9b"
    assert len(BASE_SHA)==len(V7_SHA)==64
    assert N==16 and SHARDS==4
    assert GATE["net_normalized_correct_min"]==3
    assert GATE["max_regressions"]==1
    assert source48.REPO=="framolfese/2WikiMultihopQA"


def test_frozen_existing_prompt_and_single_experimental_evidence_change():
    assert BASE_PROMPT.startswith("Answer the following question")
    assert "Return only the answer" in BASE_PROMPT
    for phrase in ("intermediate facts", "READ relevant original article",
                   "both hops", "STOP searching", "change query/source"):
        assert phrase in TREAT_PROMPT
    assert "publisher answer" not in TREAT_PROMPT.lower()
    assert "GAIA" not in TREAT_PROMPT


def test_sha_is_identical_to_original_historical_selection_protocol():
    sample=[{"hello":"中文","value":123}]
    old=sha256(json.dumps(sample,sort_keys=True).encode()).hexdigest()
    assert h(sample)==old


def test_read_counts_only_real_successful_tool_responses():
    trace=[
        TraceEvent("tool_call",1,{"id":"a","name":"search","arguments":{"query":"x"}}),
        TraceEvent("tool_result",1,{"id":"a","ok":True}),
        TraceEvent("tool_call",2,{"id":"b","name":"read","arguments":{"source":"pages/a.txt"}}),
        TraceEvent("tool_result",2,{"id":"b","ok":True}),
        TraceEvent("tool_call",3,{"id":"c","name":"read","arguments":{"source":"pages/b.txt"}}),
        TraceEvent("tool_result",3,{"id":"c","ok":False}),
        TraceEvent("tool_call",4,{"id":"d","name":"read","arguments":{"source":"pages/a.txt"}}),
        TraceEvent("tool_result",4,{"id":"d","ok":True}),
    ]
    paths,searches=read_observations(trace)
    assert paths==["pages/a.txt"]
    assert searches==1
