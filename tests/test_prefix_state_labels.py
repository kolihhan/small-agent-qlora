"""Official task-state supervision: no string-match or GAIA training labels."""
from dataclasses import replace

from gaia_small_agent.training.prefix_state_labels import PrefixEvidence, label_prefix


BASE = PrefixEvidence(
    source="appworld", split="train", steps=10, prefix_step=6,
    task_family="independent-test-family",
    source_sha256="a" * 64, trajectory_sha256="b" * 64,
    provenance_checked=True, task_family_disjoint=True,
    official_prefix_success=False, official_final_success=True,
    prefix_replayed=True, final_replayed=True, same_environment=True,
    next_action_verified=True,
)


def test_successful_long_horizon_continuation_is_continue():
    assert label_prefix(BASE).label == "CONTINUE"


def test_verified_completed_prefix_can_stop():
    assert label_prefix(replace(BASE, official_prefix_success=True)).label == "STOP"


def test_answer_string_does_not_provide_supervision():
    missing = replace(BASE, official_prefix_success=None, official_final_success=None)
    assert label_prefix(missing).label == "UNKNOWN"


def test_failed_episode_is_not_automatically_continue():
    assert label_prefix(replace(BASE, official_final_success=False)).label == "UNKNOWN"


def test_protected_or_eval_split_never_becomes_training_label():
    assert label_prefix(replace(BASE, source="GAIA-validation")).label == "UNKNOWN"
    assert label_prefix(replace(BASE, split="validation")).label == "UNKNOWN"
    assert label_prefix(replace(BASE, task_family_disjoint=False)).label == "UNKNOWN"


def test_short_nonreplayed_or_unverified_transitions_abstain():
    assert label_prefix(replace(BASE, steps=4, prefix_step=2)).label == "UNKNOWN"
    assert label_prefix(replace(BASE, prefix_replayed=False)).label == "UNKNOWN"
    assert label_prefix(replace(BASE, next_action_verified=False)).label == "UNKNOWN"
    assert label_prefix(replace(BASE, same_environment=False)).label == "UNKNOWN"
    assert label_prefix(replace(BASE, source_sha256="not-a-sha")).label == "UNKNOWN"


def test_boolean_success_contract_is_strict_not_truthiness():
    assert label_prefix(replace(BASE, official_prefix_success="false")).label == "UNKNOWN"
    assert label_prefix(replace(BASE, official_final_success=1)).label == "UNKNOWN"
