"""Pure CPU guardrail for original v7 output contract; no public gold leakage."""
from scripts.eval_2wiki16_v7_zero_call_answer_contract import (
    clean_final_caption, P, N, GATE, BASE_SHA, V7_SHA, check, h, score
)


def test_explicit_only_final_caption():
    assert clean_final_caption("My analysis.\nThe final answer is: Sunday On The Rocks")=="Sunday On The Rocks"
    assert clean_final_caption("The answer is: **Yes**")=="Yes"
    assert clean_final_caption("Final answer: Bomgay")=="Bomgay"
    assert clean_final_caption("Evil Bong")=="Evil Bong"
    assert clean_final_caption("The Answer")=="The Answer"
    assert clean_final_caption("")==""
    assert clean_final_caption("This is not a final answer.")=="This is not a final answer"


def test_no_changed_runtime_or_QLoRA_and_prespecified_gate():
    import inspect
    from scripts import eval_2wiki16_v7_zero_call_answer_contract as exp
    assert N==16 and GATE=={"min_net_gain":3,"max_regressions":0,"min_scored":16}
    assert exp.MODEL=="qwen3.5-9b"
    assert "max_steps=12" in inspect.getsource(exp.infer)
    assert "BASE_PROMPT+" in inspect.getsource(exp.infer)
    assert "gaia_score" not in inspect.getsource(exp.infer)
    assert len(BASE_SHA)==len(V7_SHA)==64


def test_synthetic_16_source_4_shards_and_zero_call_cleaner():
    from scripts.eval_2wiki16_v7_zero_call_answer_contract import score
    cases=[{"id":str(i),"question":"q","files":{},"index_titles":{}} for i in range(16)]
    src={"protocol":P,"status":"FROZEN_PRE_MODEL_SOURCE","n":N,"source_sha":h(cases),
         "model_base_sha":BASE_SHA,"model_v7_sha":V7_SHA,
         "gate":GATE,"no_gold":True,"cases":cases}
    check(src)
    references={"protocol":P,"source_sha":src["source_sha"],
                "references":[{"id":str(i),"answer":"The Name","support":[]} for i in range(N)]}
    parts=[]
    for shard in range(4):
        rows=[]
        for i in range(4*shard,4*shard+4):
            rows.append({"id":str(i),"answer":"Some thoughts.\nThe final answer is: The Name",
                         "completed":True,"stop_reason":"final",
                         "tool_calls":4,"duplicate_blocks":0,"tool_errors":0,
                         "read_paths":[]})
        parts.append({"protocol":P,"status":"ORIGINAL_V7_PREDICTIONS_SEALED",
                      "source_sha":src["source_sha"],"shard":shard,"n":4,
                      "model_base_sha":BASE_SHA,"model_v7_sha":V7_SHA,
                      "no_reference_loaded":True,"cases":rows})
    result=score(src,references,parts)
    assert result["metrics"]["old_lastline"]==0
    assert result["metrics"]["new_caption"]==16
    assert result["metrics"]["rescues"]==16
    assert result["metrics"]["regressions"]==0
    assert result["positive_screen_only"] is True
    assert result["promotion_allowed"] is False
    assert result["zero_extra_model_calls"] is True
