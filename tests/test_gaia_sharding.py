from collections import Counter

import pytest

from gaia_small_agent.benchmark.gaia100 import select_gaia_partition
from gaia_small_agent.benchmark.sharding import shard_selected_rows


def _rows():
    rows = []
    for level, count in [(1, 53), (2, 86), (3, 26)]:
        for index in range(count):
            rows.append({"task_id": f"L{level}-{index:03d}", "Level": level})
    return rows


def test_evaluation_shards_are_disjoint_complete_and_preserve_frozen_ids():
    selected = select_gaia_partition(_rows(), partition="evaluation")
    shards = [shard_selected_rows(selected, shard_index=i, shard_count=5) for i in range(5)]
    flattened = [row for shard in shards for row in shard]
    ids = [str(row["task_id"]) for row in flattened]

    assert [len(shard) for shard in shards] == [20, 20, 20, 20, 20]
    assert len(set(ids)) == 100
    assert set(ids) == {str(row["task_id"]) for row in selected}
    assert Counter(int(row["Level"]) for row in flattened) == {1: 32, 2: 52, 3: 16}


def test_shard_selection_validates_bounds():
    selected = select_gaia_partition(_rows(), partition="evaluation")
    with pytest.raises(ValueError, match="shard_count"):
        shard_selected_rows(selected, shard_index=0, shard_count=0)
    with pytest.raises(ValueError, match="shard_index"):
        shard_selected_rows(selected, shard_index=5, shard_count=5)
