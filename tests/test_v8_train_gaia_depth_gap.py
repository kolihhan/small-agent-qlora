import json
from scripts.audit_v8_train_gaia_depth_gap import audit


def test_synthetic_depth_coverage_no_answers_exported(tmp_path):
    train=tmp_path/"train.jsonl"
    train.write_text(json.dumps({
        "verified": True,
        "messages": [
            {"role":"user","content":"q"},
            {"role":"assistant","tool_calls":[{"x":1}]},
            {"role":"tool","content":"result"},
            {"role":"assistant","content":"a"}],
        "training_action_indices":[1,3]
    })+"\n")
    root=tmp_path/"gaia-v8-paired-shard-0"
    for arm,correct in (("base",False),("qlora-v8",True)):
        path=root/arm/"shard-0"/"results.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "task_id":"private","steps":2,"tool_calls":1,
            "correct":correct,"stop_reason":"final",
            "question":"never-export-me","model_answer":"secret-answer"})+"\n")
    result=audit(train,tmp_path,strict=False)
    assert result["train"]["steps"]["max"]==2
    assert result["paired_outcomes"]=={"rescue":1}
    assert result["real_model_calls"]==0
    assert "secret-answer" not in json.dumps(result)
    assert "never-export-me" not in json.dumps(result)
