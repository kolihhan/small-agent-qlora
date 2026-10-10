from __future__ import annotations

from typing import Iterable


def shard_selected_rows(rows: Iterable[dict], *, shard_index: int, shard_count: int) -> list[dict]:
    """Split an already-frozen ordered GAIA selection without changing membership."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    materialized = list(rows)
    return [row for position, row in enumerate(materialized) if position % shard_count == shard_index]


__all__ = ["shard_selected_rows"]
