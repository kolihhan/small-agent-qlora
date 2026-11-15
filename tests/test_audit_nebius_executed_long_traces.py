from scripts.audit_nebius_executed_long_traces import examine, aggregate


def fixture(n=8, success=True):
    messages=[{"role":"system","content":"OpenHands system"},
              {"role":"user","content":"Fix issue 123"}]
    for i in range(n):
        messages.append({
            "role":"assistant","content":"", "tool_calls":[{
                "id":f"call_{i}","type":"function",
                "function":{"name":"execute_bash","arguments":'{"command":"ls"}'}
            }]
        })
        messages.append({"role":"tool","tool_call_id":f"call_{i}",
                         "name":"execute_bash","content":f"Real stdout file_{i}.py"})
    messages.append({"role":"assistant","content":"Finished patch"})
    return {
        "trajectory_id":"id-one", "instance_id":"org__repo-12", "repo":"org/repo",
        "resolved":int(success),"exit_status":"submit","model_patch":"diff --git a/a b/a\n",
        "trajectory":messages
    }


def test_source_success_requires_7_real_calls_and_all_observations():
    r=examine(fixture())
    assert r["long_executed_success_candidate"] is True
    assert r["tool_calls"]==8 and r["paired_tool_observations"]==8
    assert r["assistant_tool_turns"]==8
    assert r["all_calls_observed"]
    assert not r["training_ready_for_native_gaia_policy"]


def test_missing_tool_observation_is_not_teacher_trajectory():
    x=fixture()
    x["trajectory"]=[r for r in x["trajectory"]
                     if not (r.get("role")=="tool" and r.get("tool_call_id")=="call_7")]
    r=examine(x)
    assert r["tool_calls"]==8 and r["paired_tool_observations"]==7
    assert not r["long_executed_success_candidate"]


def test_unresolved_long_trajectory_is_not_positive_sft():
    assert not examine(fixture(success=False))["long_executed_success_candidate"]


def test_short_successful_trajectory_is_not_long_horizon():
    assert not examine(fixture(n=6))["long_executed_success_candidate"]


def test_duplicate_call_id_rejected():
    x=fixture()
    x["trajectory"][4]["tool_calls"][0]["id"]="call_0"
    assert not examine(x)["long_executed_success_candidate"]


def test_aggregated_metadata_has_no_raw_agent_text_and_no_model_claim():
    x=fixture()
    r=aggregate([x],"exact-pinned-revision")
    assert r["stats"]["success_complete_seven_calls_three_turns"]==1
    assert r["native_training_examples_exported"]==0
    assert r["gpu_training_steps"]==0
    assert not r["ready_for_qlora"]
    assert r["source_example_id_hashes_only"][0]["tool_calls"]==8
    assert "Real stdout" not in str(r)
    assert "Fix issue" not in str(r)
