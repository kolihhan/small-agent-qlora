"""Ablate ONLY the known bad first-action labels in the frozen v3B design."""
from collections import Counter
import json

from tests.test_training_exposure_v3b import NativeShapeTokenizer


def test_v3e_excludes_failed_first_action_labels_while_preserving_64_step_mix(tmp_path):
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    from gaia_small_agent.training.qlora import (
        _render_turn_examples, _select_diagnostic_exposure_examples,
        _select_v3e_clean_exposure_examples,
    )
    path = generate_oracle_policy_v3_dataset(tmp_path / "oracle", seed="small-agent-policy-v3")["train"]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    examples = _render_turn_examples(NativeShapeTokenizer(), rows)
    old = _select_diagnostic_exposure_examples(examples, count=64, seed="qlora-v3b-exposure")
    selected = _select_v3e_clean_exposure_examples(examples, count=64, seed="qlora-v3b-exposure")
    assert selected == _select_v3e_clean_exposure_examples(examples, count=64, seed="qlora-v3b-exposure")
    assert len(selected) == len(old) == 64
    assert len({row["metadata"]["task_id"] for row in selected}) == 64
    assert Counter(x["metadata"]["action_type"] for x in selected) == Counter({"tool_call":35, "final":29})
    unsafe = lambda row: (
        row["metadata"]["capability"] in {"failure_recovery","strategy_switch"}
        and row["metadata"]["action_type"] == "tool_call"
        and row["metadata"]["assistant_message_index"] == 2
    )
    assert sum(unsafe(x) for x in old) == 4
    assert not any(unsafe(x) for x in selected)
    old_unaffected = {x["metadata"]["example_id"] for x in old if x["metadata"]["capability"] not in {"failure_recovery","strategy_switch"}}
    new_unaffected = {x["metadata"]["example_id"] for x in selected if x["metadata"]["capability"] not in {"failure_recovery","strategy_switch"}}
    assert old_unaffected == new_unaffected
    for cap in ("failure_recovery", "strategy_switch"):
        assert sum(x["metadata"]["capability"]==cap and x["metadata"]["action_type"]=="tool_call" for x in selected) == (5 if cap=="failure_recovery" else 4)


def test_v3e_profile_traces_do_not_regress_original_frozen_v3b_contract(tmp_path):
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    from gaia_small_agent.training.qlora import _render_turn_examples, _select_diagnostic_exposure_examples
    path = generate_oracle_policy_v3_dataset(tmp_path/"oracle")["train"]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    original = _select_diagnostic_exposure_examples(_render_turn_examples(NativeShapeTokenizer(), rows), count=64, seed="qlora-v3b-exposure")
    assert len(original)==64
