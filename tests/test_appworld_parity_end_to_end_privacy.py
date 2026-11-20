"""Boundary/privacy tests for AppWorld parity real AgentRuntime synthetic execution."""
import json
from pathlib import Path
import pytest

from gaia_small_agent.tools.appworld_parity_safe_export import (
    aggregate_private_run, validate_safe_report, write_safe_report,
    safe_exception_category,
)
from scripts.appworld_parity_synthetic_privacy_smoke import (
    _SECRET, exercise_synthetic_private_workflow, run_smoke,
)


def test_private_full_trace_and_recorder_do_not_enter_safe_report(tmp_path):
    row=exercise_synthetic_private_workflow(tmp_path/"first")
    validate_safe_report(row)
    assert row["arm"]=="parity"
    assert row["agent_metrics"]["tool_calls"]==4
    assert row["actual_api_calls"]==4
    assert row["auth_action_counts"]["supervisor_passwords"]["attempted"]==1
    assert row["auth_action_counts"]["app_login"]["succeeded"]==1
    assert row["http_status_histogram"]=={}
    assert row["raw_model_recording_exported"] is False
    assert row["raw_agent_trace_exported"] is False
    dump=json.dumps(row)
    assert all(s not in dump for s in _SECRET)


def test_secret_carrying_model_error_message_is_not_serialized(tmp_path):
    row=exercise_synthetic_private_workflow(tmp_path/"failure",fail_after_password=True)
    assert row["stop_reason"]=="model_error"
    assert row["auth_action_counts"]["supervisor_passwords"]["succeeded"]==1
    assert row["auth_action_counts"]["app_login"]["attempted"]==0
    assert all(s not in json.dumps(row) for s in _SECRET)


def test_safe_artifact_writer_accepts_only_explicit_schema_and_is_immutable(tmp_path):
    row=exercise_synthetic_private_workflow(tmp_path/"model")
    path=tmp_path/"safe.json"
    write_safe_report(path,row)
    assert all(s not in path.read_text() for s in _SECRET)
    with pytest.raises(FileExistsError):
        write_safe_report(path,row)
    bad=dict(row)
    bad["raw_trace"]={"tool_result":"SYNTHETIC-PASSWORD-PRIVATE-SENTINEL-90823"}
    with pytest.raises(ValueError,match="unexpected aggregate"):
        write_safe_report(tmp_path/"not-saved.json",bad)
    assert not (tmp_path/"not-saved.json").exists()
    bad2=dict(row)
    bad2["stop_reason"]="private user secret"
    with pytest.raises(ValueError,match="unsafe stop"):
        validate_safe_report(bad2)
    bad3=dict(row)
    bad3["http_status_histogram"]={"SYNTHETIC-PASSWORD-PRIVATE-SENTINEL-90823":1}
    with pytest.raises(ValueError,match="unsafe status"):
        validate_safe_report(bad3)


def test_safe_exception_emission_rejects_exception_messages():
    e=ValueError("SYNTHETIC-PASSWORD-PRIVATE-SENTINEL-90823")
    assert safe_exception_category(e)=="INVALID_DATA"
    assert all(s not in safe_exception_category(e) for s in _SECRET)


def test_cli_like_synthetic_smoke_has_only_sentinel_free_artifact_and_stdout(
    tmp_path,capsys
):
    output=tmp_path/"safe-output.json"
    run_smoke(output)
    text=capsys.readouterr()
    assert all(s not in text.out+text.err+output.read_text() for s in _SECRET)
    report=json.loads(output.read_text())
    assert report["n_synthetic_scenarios"]==2
    assert len(report["safe_agent_reports"])==2
    assert report["safe_agent_reports"][1]["stop_reason"]=="model_error"
    assert list(tmp_path.iterdir())==[output]
