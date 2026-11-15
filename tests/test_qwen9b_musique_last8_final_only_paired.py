from collections import Counter

import pytest

import scripts.qwen9b_musique_last8_final_only_paired as exp
from gaia_small_agent.agent.loop import SYSTEM_PROMPT


def _case(i,base,final):
    hop=3 if i<4 else 4
    def arm(correct):
        return {
            "exact_correct":correct,
            "completed":True,
            "tool_calls":4,
            "tool_errors":0,
            "gold_literal_in_output":correct,
            "latency_s":2.5,
        }
    return {"task_id":f"last-{i}","hop_count":hop,
            "baseline":arm(base),"final_only":arm(final)}


def _shards():
    cases=[_case(i,i in (0,4),i in (0,2,4,6)) for i in range(8)]
    return [{"shard":j,"protocol":exp.PROTOCOL,
             "model_digest":exp.EXPECTED_MODEL_DIGEST,
             "task_selection_sha256":"frozen",
             "source_manifest_sha256":exp.EXPECTED_MANIFEST,
             "cases":cases[2*j:2*j+2]} for j in range(4)]


def test_pre_registered_exact_success_rescue_regression_gate():
    shards=_shards()
    report=exp.aggregate(shards)
    assert (report["baseline_exact"],report["final_only_exact"])==(2,4)
    assert (report["rescues"],report["regressions"])==(2,0)
    assert report["rescued_hops"]==[3,4]
    assert report["positive_screen_only"] is True
    assert report["promotion_allowed"] is False
    assert report["not_GAIA_or_matched_v8"] is True
    shards[0]["cases"][0]["final_only"]["exact_correct"]=False
    fail=exp.aggregate(shards)
    assert fail["positive_screen_only"] is False
    assert fail["regressions"]==1


def test_all_eight_heldout_questions_and_model_version_are_required():
    shards=_shards()
    with pytest.raises(ValueError,match="four paired shards"):
        exp.aggregate(shards[:3])
    shards[1]["model_digest"]="wrong"
    with pytest.raises(ValueError,match="four paired shards"):
        exp.aggregate(shards)


def test_only_generic_final_instruction_and_pinned_original_model():
    assert exp.FINAL_ONLY_PROMPT.startswith(SYSTEM_PROMPT)
    assert "ONLY the shortest factual" in exp.FINAL_ONLY_PROMPT
    assert exp.MODEL=="qwen3.5:9b-q4_K_M"
    assert exp.freeze_last is not None
