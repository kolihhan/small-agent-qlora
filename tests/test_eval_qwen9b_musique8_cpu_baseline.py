import pytest
from scripts.eval_qwen9b_musique8_cpu_baseline import (
    PROTOCOL, EXPECTED_MODEL_DIGEST, aggregate, freeze, ollama_model_digest,
)


def test_aggregate_real_closed_loop_requires_complete_hop_balance():
    shards = []
    for i in range(4):
        cases=[{
            "task_id":f"parent-{2*i+j}",
            "hop_count":3 if i<2 else 4,
            "completed":True,
            "correct":j==0,
            "tool_calls":5,"tool_errors":0,
            "duplicate_calls_blocked":1,
            "stop_reason":"final","latency_s":6.5,
        } for j in range(2)]
        shards.append({"protocol":PROTOCOL,"shard":i,
                       "task_selection_hash":"frozen-sample", "cohort_sha256":"frozen-physical",
                       "model_digest":EXPECTED_MODEL_DIGEST,"cases":cases})
    result=aggregate(shards)
    assert result["exact_correct"]==4 and result["total"]==8
    assert result["by_hop"]=={"3":{"total":4,"correct":2},"4":{"total":4,"correct":2}}
    assert result["tool_calls"]==40 and result["duplicates"]==8
    assert result["is_matched_v8_eval"] is False and result["promotion_allowed"] is False
    with pytest.raises(ValueError,match="four"):
        aggregate(shards[:3])
    shards[0]["model_digest"]="wrong"
    with pytest.raises(ValueError,match="four"):
        aggregate(shards)


def test_cannot_change_frozen_24_parent_provenance():
    with pytest.raises(ValueError,match="physical MuSiQue24"):
        freeze([],{})


def test_model_digest_check_not_just_model_name(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"models":[{"name":"qwen3.5:9b-q4_K_M","digest":"wrong"}]}
    monkeypatch.setattr("requests.get",lambda *_a,**_kw:Response())
    with pytest.raises(ValueError,match="digest differs"):
        ollama_model_digest("http://localhost:11434","qwen3.5:9b-q4_K_M")
