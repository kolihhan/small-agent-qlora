from __future__ import annotations

from typing import Any


V3D_CPU_BATCH_SIZE = 8
V3D_CPU_BATCH_COUNT = 3
V3D_EXPECTED_EXAMPLE_COUNT = V3D_CPU_BATCH_SIZE * V3D_CPU_BATCH_COUNT


def partition_v3d_examples(
    examples: list[dict[str, Any]],
    *,
    batch_size: int = V3D_CPU_BATCH_SIZE,
) -> list[list[dict[str, Any]]]:
    """Partition the frozen 24-example v3D order into resumable CPU batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(examples) != V3D_EXPECTED_EXAMPLE_COUNT:
        raise ValueError(
            f"v3D CPU continuation expects exactly {V3D_EXPECTED_EXAMPLE_COUNT} examples, "
            f"got {len(examples)}"
        )
    batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
    if len(batches) != V3D_CPU_BATCH_COUNT or any(len(batch) != batch_size for batch in batches):
        raise ValueError(
            f"v3D CPU continuation must split into exactly {V3D_CPU_BATCH_COUNT} "
            f"equal batches of {batch_size}"
        )
    return batches


def expected_completed_example_ids(
    examples: list[dict[str, Any]],
    *,
    completed_batch_count: int,
    batch_size: int = V3D_CPU_BATCH_SIZE,
) -> list[str]:
    """Return the exact frozen prefix that must have completed before a resume."""
    if not 0 <= completed_batch_count <= V3D_CPU_BATCH_COUNT:
        raise ValueError("completed_batch_count is out of range")
    partition_v3d_examples(examples, batch_size=batch_size)
    completed = examples[: completed_batch_count * batch_size]
    return [str(item["metadata"]["example_id"]) for item in completed]


__all__ = [
    "V3D_CPU_BATCH_COUNT",
    "V3D_CPU_BATCH_SIZE",
    "V3D_EXPECTED_EXAMPLE_COUNT",
    "expected_completed_example_ids",
    "partition_v3d_examples",
]
