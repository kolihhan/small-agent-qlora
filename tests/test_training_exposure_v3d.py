from collections import Counter

import pytest


def _candidate(capability: str, action_type: str, message_index: int, number: int) -> dict:
    task_id = f"train-{capability}-{message_index}-{number:03d}"
    return {
        "prompt": f"prompt {task_id}",
        "completion": f"completion {task_id}",
        "metadata": {
            "example_id": f"{task_id}:{message_index}",
            "task_id": task_id,
            "capability": capability,
            "action_type": action_type,
            "assistant_message_index": message_index,
        },
    }


def _pool() -> list[dict]:
    from gaia_small_agent.training.exposure_v3d import V3D_STATE_QUOTAS

    rows = []
    for (capability, action_type, message_index), quota in V3D_STATE_QUOTAS.items():
        for number in range(quota + 6):
            rows.append(_candidate(capability, action_type, message_index, number))

    # These are deliberately poisoned initial actions from the v3 oracle.  They
    # exist in the candidate pool to prove the selector will never pick them.
    for capability in ("failure_recovery", "strategy_switch"):
        for number in range(12):
            rows.append(_candidate(capability, "tool_call", 2, 100 + number))
    return rows


def test_v3d_selects_24_state_aware_corrective_decisions_deterministically():
    from gaia_small_agent.training.exposure_v3d import (
        V3D_STATE_QUOTAS,
        select_v3d_corrective_examples,
        summarize_v3d_selection,
    )

    pool = _pool()
    selected = select_v3d_corrective_examples(pool, seed="qlora-v3d-corrective")
    repeated = select_v3d_corrective_examples(pool, seed="qlora-v3d-corrective")

    assert [item["metadata"]["example_id"] for item in selected] == [
        item["metadata"]["example_id"] for item in repeated
    ]
    assert len(selected) == sum(V3D_STATE_QUOTAS.values()) == 24
    assert len({item["metadata"]["task_id"] for item in selected}) == 24

    report = summarize_v3d_selection(selected)
    assert report["selected_example_count"] == 24
    assert report["action_type_counts"] == {"final": 12, "tool_call": 12}


def test_v3d_never_teaches_initial_failure_as_recovery_or_switch():
    from gaia_small_agent.training.exposure_v3d import select_v3d_corrective_examples

    selected = select_v3d_corrective_examples(_pool(), seed="qlora-v3d-corrective")
    for item in selected:
        meta = item["metadata"]
        capability = meta["capability"]
        if capability in {"failure_recovery", "strategy_switch"} and meta["action_type"] == "tool_call":
            assert meta["assistant_message_index"] == 4
        if capability in {"failure_recovery", "strategy_switch"} and meta["action_type"] == "final":
            assert meta["assistant_message_index"] == 6

    poisoned = {
        item["metadata"]["example_id"]
        for item in _pool()
        if item["metadata"]["capability"] in {"failure_recovery", "strategy_switch"}
        and item["metadata"]["assistant_message_index"] == 2
    }
    assert not poisoned.intersection(item["metadata"]["example_id"] for item in selected)


def test_v3d_preserves_the_frozen_corrective_mix():
    from gaia_small_agent.training.exposure_v3d import select_v3d_corrective_examples

    selected = select_v3d_corrective_examples(_pool(), seed="qlora-v3d-corrective")
    counts = Counter(
        (
            item["metadata"]["capability"],
            item["metadata"]["action_type"],
            item["metadata"]["assistant_message_index"],
        )
        for item in selected
    )
    assert counts == Counter(
        {
            ("evidence_to_final", "final", 4): 4,
            ("multi_step", "tool_call", 4): 2,
            ("multi_step", "final", 6): 3,
            ("failure_recovery", "tool_call", 4): 3,
            ("failure_recovery", "final", 6): 3,
            ("strategy_switch", "tool_call", 4): 2,
            ("strategy_switch", "final", 6): 1,
            ("argument_grounding", "tool_call", 2): 3,
            ("tool_selection", "tool_call", 2): 1,
            ("duplicate_avoidance", "tool_call", 2): 1,
            ("no_tool_stop", "final", 2): 1,
        }
    )


def test_v3d_can_exclude_tasks_already_seen_by_v3b():
    from gaia_small_agent.training.exposure_v3d import select_v3d_corrective_examples

    pool = _pool()
    first = select_v3d_corrective_examples(pool, seed="qlora-v3d-corrective")
    excluded = {item["metadata"]["task_id"] for item in first}
    second = select_v3d_corrective_examples(
        pool,
        seed="qlora-v3d-corrective",
        exclude_task_ids=excluded,
    )
    assert excluded.isdisjoint(item["metadata"]["task_id"] for item in second)
    assert len(second) == 24


def test_v3d_fails_closed_when_a_required_state_is_missing():
    from gaia_small_agent.training.exposure_v3d import select_v3d_corrective_examples

    pool = [
        item
        for item in _pool()
        if not (
            item["metadata"]["capability"] == "failure_recovery"
            and item["metadata"]["action_type"] == "tool_call"
            and item["metadata"]["assistant_message_index"] == 4
        )
    ]
    with pytest.raises(ValueError, match="failure_recovery/tool_call/index4"):
        select_v3d_corrective_examples(pool, seed="qlora-v3d-corrective")
