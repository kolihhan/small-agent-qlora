from scripts.audit_nextsearch_apache_source import inspect, audit


def task(n=8, answer="Blue"):
    target=[]
    for i in range(n):
        tool="search" if i%2==0 else "fetch"
        ident=f"call_{i}"
        target.extend([
            {"role":"assistant","content":"","tool_calls":[{
                "id":ident,"type":"function","function":{
                    "name":tool,"arguments":'{"query":"the original question"}'
                }
            }]},
            {"role":"tool","tool_call_id":ident,"content":"Original visible web source evidence"}
        ])
    target.append({"role":"assistant","content":answer,"tool_calls":None})
    return {
        "id":"taskcraft/test-1","source":"taskcraft","license":"apache-2.0",
        "band":"hard","corpus":"s-xs","gold":{"answer":"Blue"},
        "messages":[{"role":"system","content":"research"},{"role":"user","content":"question"}],
        "tools":[{"name":"search"},{"name":"fetch"}],"target":target
    }


def test_valid_eight_real_calls_with_credible_observations_and_gold_answer():
    out=inspect(task())
    assert out["all_actual_calls_observed"] is True
    assert out["tool_actions"]==8
    assert out["matched_nonempty_observations"]==8
    assert out["assistant_action_turns"]==8
    assert out["strict_gold_answer_exact"]
    assert out["long_search_fetch_candidate"]
    assert out["tool_names"]==["fetch","search"]
    assert not out["tool_schema_native_to_our_gaia"]


def test_missing_feedback_rejects_even_if_gold_answer_correct():
    row=task()
    row["target"].pop(5)
    out=inspect(row)
    assert not out["all_actual_calls_observed"]
    assert not out["long_search_fetch_candidate"]


def test_wrong_gold_does_not_create_supervised_teacher_success():
    out=inspect(task(answer="Red"))
    assert out["all_actual_calls_observed"] and not out["strict_gold_answer_exact"]
    assert not out["long_search_fetch_candidate"]


def test_no_pseudo_tool_trajectory_and_no_jumping_to_final():
    assert not inspect(task(n=3))["long_search_fetch_candidate"]
    nofetch=task()
    for msg in nofetch["target"]:
        if msg["role"]=="assistant" and msg.get("tool_calls"):
            msg["tool_calls"][0]["function"]["name"]="search"
    assert not inspect(nofetch)["long_search_fetch_candidate"]


def test_original_source_permission_and_task_id_disjoint_are_mandatory():
    row=task()
    summary=audit([row],"pinned-source-commit")
    assert summary["stats"]["long_search_fetch_gold_exact"]==1
    assert summary["stats"]["long_candidate_real_actions"]==8
    assert summary["ready_for_qlora"] is False
    assert summary["training_data_exported"]==0
    assert "Original visible web source" not in str(summary)
    bad=task()
    bad["license"]="cc-by-nc-4.0"
    import pytest
    with pytest.raises(ValueError,match="nonpermissive"):
        audit([bad],"sha")
    dup=task()
    paired=audit([row,dup],"sha")
    assert paired["stats"]["duplicate_task_id_rows"]==1
    assert paired["unique_task_ids"]==1
    assert paired["unique_long_gold_exact_task_ids"]==1


def test_malformed_tool_arguments_or_call_order_rejected():
    row=task()
    row["target"][0]["tool_calls"][0]["function"]["arguments"]="{not json"
    assert not inspect(row)["all_actual_calls_observed"]
    row2=task()
    row2["target"][1]["tool_call_id"]="not_existing"
    assert not inspect(row2)["all_actual_calls_observed"]
