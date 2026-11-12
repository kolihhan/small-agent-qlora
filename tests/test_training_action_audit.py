import pytest

from scripts.audit_v3b_training_labels import audit_v3b_selected_actions
from scripts.prepare_teacher_trials import prepare_trials


def test_v3b_audit_is_deterministic_and_identifies_supervised_bad_actions():
    report = audit_v3b_selected_actions()
    assert report == audit_v3b_selected_actions()
    assert report["selected"] == 64
    assert report["unsafe_failed_first_actions"] > 0
    assert set(report["unsafe_by_capability"]) <= {"failure_recovery", "strategy_switch"}
    assert len(report["unsafe_action_ids"]) == report["unsafe_failed_first_actions"]


def test_repeated_trials_are_groupable_and_deterministic():
    task = {
        "id": "independent-task", "source": "synthetic-tool-domain",
        "question": "Find the latest entry in the local ledger.",
        "expected_answer": "A", "license": "CC0-1.0",
        "generator": "unit", "generator_version": "1",
        "oracle_type": "exact", "oracle_version": "1",
        "required_tools": ["read"],
    }
    result = prepare_trials([task], samples=3)
    assert [row["id"] for row in result] == [
        "independent-task::sample-1",
        "independent-task::sample-2",
        "independent-task::sample-3",
    ]
    assert all(row["question"] == task["question"] for row in result)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_trials([task, task], samples=3)
    with pytest.raises(ValueError, match="GAIA"):
        prepare_trials([{**task, "source": "GAIA-validation"}], samples=3)
    with pytest.raises(ValueError, match="range"):
        prepare_trials([task], samples=1)
