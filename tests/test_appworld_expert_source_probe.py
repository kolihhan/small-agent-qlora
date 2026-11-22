"""Source-only AppWorld expert feasibility: no gold leakage or training claims."""
from scripts.appworld_expert_source_probe import select_train_tasks, safe_summary


def test_source_selection_is_deterministic_scene_disjoint():
    roster = ["sc1_1", "sc1_2", "sc2_1", "sc3_1", "sc4_2", "sc5_1"]
    a = select_train_tasks(roster, count=4)
    b = select_train_tasks(list(reversed(roster)), count=4)
    assert a == b
    assert len(a) == 4
    assert len({task.split("_", 1)[0] for task in a}) == 4


def test_expert_success_is_not_mistaken_for_native_long_agent_supervision():
    rows = [
        {"official_expert_success": True, "publisher_api_call_count": 9},
        {"official_expert_success": False, "publisher_api_call_count": 4},
    ]
    out = safe_summary(rows, source_sha256="b" * 64, sample_sha256="a" * 64)
    assert out["official_expert_terminal_success"] == 1
    assert out["publisher_declared_api_depth_at_least_7"] == 1
    assert out["verified_native_agent_long_horizon_success"] == 0
    assert out["native_stop_labels"] == {"STOP": 0, "CONTINUE": 0, "UNKNOWN": 2}
    assert "rows" not in out and "task_ids" not in out


def test_bad_official_success_type_fails_closed():
    import pytest
    with pytest.raises(ValueError):
        safe_summary(
            [{"official_expert_success": "true", "publisher_api_call_count": 8}],
            source_sha256="b"*64, sample_sha256="a"*64,
        )


def test_worker_output_accepts_only_safe_schema_and_never_returns_raw_logs():
    from scripts.appworld_expert_source_probe import parse_safe_worker_output
    safe = safe_summary(
        [{"official_expert_success":True,"publisher_api_call_count":9}],
        source_sha256="b"*64, sample_sha256="a"*64,
    )
    import json
    line = "APPWORLD_SAFE_AGGREGATE=" + json.dumps(safe)
    assert parse_safe_worker_output("potential private SDK log ignored\n"+line)==safe

    import pytest
    with pytest.raises(ValueError):
        parse_safe_worker_output("potential private SDK log ignored\n")
    with pytest.raises(ValueError):
        parse_safe_worker_output(line+"\n"+line)
