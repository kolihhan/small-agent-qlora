from __future__ import annotations

import pytest

from gaia_small_agent.training.exposure_v3d import plan_v3d_batches


def _selected(count: int = 24) -> list[dict]:
    return [
        {
            "metadata": {
                "example_id": f"example-{index:02d}",
                "task_id": f"task-{index:02d}",
                "capability": "multi_step",
                "action_type": "tool_call" if index % 2 else "final",
                "assistant_message_index": 4,
            }
        }
        for index in range(count)
    ]


def test_plan_v3d_batches_preserves_training_order_and_exact_coverage() -> None:
    selected = _selected()
    batches = plan_v3d_batches(selected, batch_size=6)

    assert len(batches) == 4
    assert [len(batch) for batch in batches] == [6, 6, 6, 6]
    flattened = [item["metadata"]["example_id"] for batch in batches for item in batch]
    assert flattened == [item["metadata"]["example_id"] for item in selected]
    assert len(flattened) == len(set(flattened)) == 24


def test_plan_v3d_batches_allows_final_partial_batch() -> None:
    batches = plan_v3d_batches(_selected(10), batch_size=4)
    assert [len(batch) for batch in batches] == [4, 4, 2]


def test_plan_v3d_batches_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        plan_v3d_batches(_selected(), batch_size=0)
