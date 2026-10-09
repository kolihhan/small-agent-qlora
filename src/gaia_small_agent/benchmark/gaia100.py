from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable


QUOTAS = {1: 32, 2: 52, 3: 16}
DIAGNOSTIC_QUOTAS = {1: 8, 2: 13, 3: 4}
SHADOW_QUOTAS = {1: 8, 2: 13, 3: 4}
LOCAL_PROTOCOL_SEED = "gaia-local-v2"
GAIA_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"


def _rank(task_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def select_gaia_partition(rows: Iterable[dict], partition: str, seed: str = LOCAL_PROTOCOL_SEED) -> list[dict]:
    """Select a deterministic, disjoint local GAIA partition.

    Diagnostic rows are allocated first within each level. Evaluation rows use
    the next ranked rows. Shadow rows are selected from the remaining pool with
    a separate seed namespace so existing Diagnostic25/Evaluation100 identities
    stay unchanged.
    """
    if partition not in {"diagnostic", "evaluation", "shadow"}:
        raise ValueError("partition must be 'diagnostic', 'shadow', or 'evaluation'")
    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_level[int(row["Level"])].append(row)
    selected: list[dict] = []
    for level in (1, 2, 3):
        pool = sorted(by_level[level], key=lambda r: (_rank(str(r["task_id"]), seed), str(r["task_id"])))
        diagnostic_quota = DIAGNOSTIC_QUOTAS[level]
        evaluation_quota = QUOTAS[level]
        reserved = diagnostic_quota + evaluation_quota
        if partition == "shadow":
            shadow_quota = SHADOW_QUOTAS[level]
            if len(pool) < reserved + shadow_quota:
                raise ValueError(
                    f"GAIA level {level} has {len(pool)} rows, need {reserved + shadow_quota} "
                    "for diagnostic, evaluation, and shadow"
                )
            remaining = pool[reserved:]
            shadow_seed = f"{seed}:shadow"
            shadow_pool = sorted(
                remaining,
                key=lambda r: (_rank(str(r["task_id"]), shadow_seed), str(r["task_id"])),
            )
            selected.extend(shadow_pool[:shadow_quota])
            continue

        if len(pool) < reserved:
            raise ValueError(f"GAIA level {level} has {len(pool)} rows, need {reserved}")
        start = 0 if partition == "diagnostic" else diagnostic_quota
        quota = diagnostic_quota if partition == "diagnostic" else evaluation_quota
        selected.extend(pool[start:start + quota])

    order_seed = f"{seed}:shadow" if partition == "shadow" else seed
    return sorted(selected, key=lambda r: (int(r["Level"]), _rank(str(r["task_id"]), order_seed)))


def select_gaia100(rows: Iterable[dict], seed: str = "gaia100-v1") -> list[dict]:
    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_level[int(row["Level"])].append(row)
    selected: list[dict] = []
    for level, quota in QUOTAS.items():
        pool = sorted(by_level[level], key=lambda r: (_rank(str(r["task_id"]), seed), str(r["task_id"])))
        if len(pool) < quota:
            raise ValueError(f"GAIA level {level} has {len(pool)} rows, need {quota}")
        selected.extend(pool[:quota])
    return sorted(selected, key=lambda r: (int(r["Level"]), _rank(str(r["task_id"]), seed)))


def resolve_file_paths(rows: Iterable[dict], data_dir: str | Path) -> list[dict]:
    root = Path(data_dir).resolve()
    resolved = []
    for row in rows:
        item = dict(row)
        raw = item.get("file_path")
        if raw:
            path = Path(str(raw))
            if not path.is_absolute():
                path = root / path
            item["file_path"] = str(path.resolve())
        resolved.append(item)
    return resolved


def load_validation(source: str = "gaia-benchmark/GAIA", token: str | None = None, revision: str = GAIA_REVISION):
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc

    token = token or os.getenv("HF_TOKEN")
    if Path(source).exists():
        data_dir = str(Path(source).resolve())
    else:
        data_dir = snapshot_download(repo_id=source, repo_type="dataset", token=token, revision=revision)
    dataset = load_dataset(data_dir, "2023_all", split="validation")
    return resolve_file_paths(dataset, data_dir)


def _population_summary(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def summarize_results(rows: list[dict]) -> dict:
    from collections import Counter

    total = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    completed = sum(bool(r["completed"]) for r in rows)
    calls = sum(int(r.get("tool_calls", 0)) for r in rows)
    successes = sum(int(r.get("tool_successes", 0)) for r in rows)
    levels = {}
    for level in (1, 2, 3):
        part = [r for r in rows if int(r["level"]) == level]
        levels[str(level)] = _population_summary(part)
    supported = [r for r in rows if not r.get("capability_gap")]
    capability_gap = [r for r in rows if r.get("capability_gap")]
    failure_labels = Counter(str(r.get("failure_label") or "unclassified") for r in rows)
    failure_signals = Counter(signal for r in rows for signal in r.get("failure_signals", []))
    return {
        "name": "GAIA local evaluation",
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "completed": completed,
        "completion_rate": completed / total if total else 0.0,
        "tool_calls": calls,
        "tool_successes": successes,
        "tool_errors": sum(int(r.get("tool_errors", 0)) for r in rows),
        "duplicate_calls_blocked": sum(int(r.get("duplicate_calls_blocked", 0)) for r in rows),
        "tool_success_rate": successes / calls if calls else 1.0,
        "average_steps": sum(int(r.get("steps", 0)) for r in rows) / total if total else 0.0,
        "model_calls": sum(int(r.get("model_calls", r.get("steps", 0))) for r in rows),
        "mean_latency_ms": sum(float(r.get("latency_ms", 0.0)) for r in rows) / total if total else 0.0,
        "levels": levels,
        "supported": _population_summary(supported),
        "capability_gap": _population_summary(capability_gap),
        "failure_labels": dict(sorted(failure_labels.items())),
        "failure_signals": dict(sorted(failure_signals.items())),
    }


def save_local_manifest(
    selected: list[dict],
    path: str | Path,
    seed: str,
    dataset_revision: str = GAIA_REVISION,
    partition: str = "evaluation",
) -> None:
    path = Path(path)
    quotas_by_partition = {
        "diagnostic": DIAGNOSTIC_QUOTAS,
        "shadow": SHADOW_QUOTAS,
        "evaluation": QUOTAS,
    }
    if partition not in quotas_by_partition:
        raise ValueError("partition must be 'diagnostic', 'shadow', or 'evaluation'")
    quotas = quotas_by_partition[partition]
    data = {
        "name": f"GAIA local {partition} partition",
        "partition": partition,
        "seed": seed,
        "dataset_revision": dataset_revision,
        "quotas": {str(level): quota for level, quota in quotas.items()},
        "task_ids": [str(r["task_id"]) for r in selected],
        "note": "Local-only manifest. Do not publish GAIA gated content.",
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("GAIA manifest mismatch: existing manifest is unreadable") from exc
        if existing != data:
            raise ValueError("GAIA manifest mismatch: existing manifest does not match requested subset")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
