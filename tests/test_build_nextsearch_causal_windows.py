import json

from gaia_small_agent.training.qlora import load_verified_rows
from scripts.build_nextsearch_causal_windows import state_windows,build


def parent_task(early_answer=False):
    messages=[
        {"role":"system","content":"Source search"},
        {"role":"user","content":"Identify an independently verifiable source fact"}
    ]
    for i in range(8):
        tool="search" if i%2==0 else "fetch"
        messages.extend([
            {"role":"assistant","content":"","tool_calls":[{
                "id":f"call_{i}","type":"function",
                "function":{"name":tool,"arguments":{"objective":"Find source"}}
            }]},
            {"role":"tool","tool_call_id":f"call_{i}","name":tool,
             "content":("The answer is Verified Blue."
                        if (early_answer and i==0) or (not early_answer and i==7)
                        else "Original source webpage text "+str(i))}
        ])
    messages.append({"role":"assistant","content":"Verified Blue"})
    return {
        "task_id":"taskcraft/original-source-117",
        "source":"NextTokenAI/NextSearch-1-Trajectories",
        "source_split":"train",
        "pilot_only":True,
        "capability":"evidence_conditioned_multistep_web_research",
        "verified":True,
        "verification":{
            "completed":True,"final_answer_correct":True,
            "zero_tool_errors":True,"zero_duplicate_blocks":True,
            "required_tools_satisfied":True,"provenance_complete":True,
        },
        "provenance":{
            "license":"Apache-2.0","generator":"NextTokenAI/NextSearch-1-Trajectories",
            "generator_version":"pinned","oracle_type":"original-source",
            "oracle_version":"v1",
        },
        "required_tools":["search","fetch"],
        "expected_answer":"Verified Blue",
        "tools":[{"type":"function","function":{
            "name":"search","description":"Search",
            "parameters":{"type":"object","properties":{}}}}],
        "messages":messages,
        "metrics":{"steps":8,"tool_calls":8,"tool_errors":0}
    }


def test_real_original_actions_produce_missing_and_complete_source_state():
    parent=parent_task()
    parts,report=state_windows(parent)
    assert report["missing_state"]==1
    assert report["complete_state"]==1
    assert len(parts)==2
    missing,nextpart=parts
    assert missing["capability"]=="evidence_missing_continue"
    assert not missing["source_answer_visible_in_window"]
    assert missing["messages"][-1]["tool_calls"][0]["function"]["name"]=="search"
    assert nextpart["capability"]=="evidence_present_finalize"
    assert nextpart["source_answer_visible_in_window"]
    assert nextpart["messages"][-1]==parent["messages"][-1]
    assert all(x["original_teacher_action_unchanged"] for x in parts)
    assert all(x["source_context_is_partial"] for x in parts)
    assert all(x["task_success_independently_replayed"] is False for x in parts)
    assert len(missing["messages"]) < len(parent["messages"])
    assert len(nextpart["messages"]) < len(parent["messages"])


def test_early_answer_discards_oversearch_training_target():
    parts,stats=state_windows(parent_task(early_answer=True))
    assert not parts
    assert stats["oversearched_after_evidence"]==1


def test_original_loader_accepts_real_partial_state_labels_without_gaia(tmp_path):
    output=tmp_path/"windowed.jsonl"
    report=build([parent_task()],output)
    assert report["actual_partial_source_state_training_rows"]==2
    assert report["parents_with_both_states"]==1
    assert len(load_verified_rows(output))==2


def test_duplicated_source_parents_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError,match="duplicate"):
        build([parent_task(),parent_task()],tmp_path/"output.jsonl")
