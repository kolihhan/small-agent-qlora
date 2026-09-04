from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

from ..agent.loop import AgentRuntime
from ..tools.content_type import detect_content_type
from .gaia100 import GAIA_REVISION, save_local_manifest, select_gaia_partition, summarize_results
from .scoring import gaia_score


_RESULT_REQUIRED_KEYS = frozenset(
    {
        "index",
        "task_id",
        "level",
        "model_answer",
        "correct",
        "completed",
        "stop_reason",
        "steps",
        "tool_calls",
        "tool_successes",
        "tool_errors",
        "duplicate_calls_blocked",
        "capability_gap",
        "failure_label",
        "trace",
    }
)


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def known_capability_gap(file_path: str | Path | None) -> str | None:
    if not file_path:
        return None
    path = Path(str(file_path))
    suffix = path.suffix.casefold()
    detected = detect_content_type(path) if path.is_file() else "unknown"
    if suffix in _IMAGE_SUFFIXES or detected == "image":
        return "semantic_image_unsupported"
    if suffix in _AUDIO_SUFFIXES or detected == "audio":
        return "semantic_audio_unsupported"
    if suffix in _VIDEO_SUFFIXES or detected == "video":
        return "semantic_video_unsupported"
    return None


def classify_failure_label(result, *, correct: bool, capability_gap: str | None) -> str:
    # This is a transparent rule-based diagnostic label, not a causal root-cause claim.
    if correct:
        return "correct"
    if capability_gap:
        return "capability_gap"
    if result.stop_reason == "model_capacity":
        return "model_capacity"
    if result.stop_reason == "max_steps":
        return "max_steps"
    if result.stop_reason == "empty_final":
        return "empty_final"
    if result.metrics.duplicate_calls_blocked:
        return "duplicate_action_present"
    if result.metrics.tool_errors:
        return "tool_error_present"
    if result.completed and result.metrics.tool_calls == 0:
        return "premature_final_proxy"
    if result.completed:
        return "incorrect_after_tools"
    return "incomplete_other"




def collect_failure_signals(result, *, correct: bool, capability_gap: str | None) -> list[str]:
    if correct:
        return []
    signals: list[str] = []
    if result.stop_reason == "max_steps":
        signals.append("max_steps")
    if result.metrics.duplicate_calls_blocked:
        signals.append("duplicate_block")
    if result.metrics.tool_errors:
        signals.append("tool_error")
    if capability_gap:
        signals.append("capability_gap")
    if result.stop_reason == "empty_final":
        signals.append("malformed_output")
    if result.stop_reason == "model_capacity":
        signals.append("model_capacity")
    if result.completed and result.metrics.tool_calls == 0:
        signals.append("premature_final_proxy")
    return signals

def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _ensure_run_config(work_root: Path, results_path: Path, run_config: dict | None = None) -> None:
    config_path = work_root / "run_config.json"
    results_nonempty = results_path.exists() and results_path.stat().st_size > 0
    mismatch = "GAIA run configuration mismatch: existing configuration does not match requested run"
    if run_config is None:
        if results_nonempty or config_path.exists():
            raise ValueError(mismatch)
        return
    if not isinstance(run_config, dict):
        raise ValueError(mismatch)
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(mismatch) from exc
        if existing != run_config:
            raise ValueError(mismatch)
        return
    if results_nonempty:
        raise ValueError(mismatch)
    _atomic_write_text(config_path, json.dumps(run_config, indent=2, sort_keys=True))


def _load_existing_results(results_path: Path, manifest_task_ids: list[str]) -> dict[str, dict]:
    """Load persisted records, repairing only an interrupted final JSON line."""
    if not results_path.exists() or results_path.stat().st_size == 0:
        return {}
    try:
        raw = results_path.read_bytes()
        lines = re.split(r"\r\n|\n|\r", raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("GAIA results are unreadable") from exc

    nonempty = [line for line in lines if line.strip()]
    newline_terminated = raw.endswith((b"\n", b"\r"))
    allowed_ids = set(manifest_task_ids)
    records: dict[str, dict] = {}
    valid_lines: list[str] = []
    discarded_final_fragment = False
    for position, line in enumerate(nonempty):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if position == len(nonempty) - 1 and not newline_terminated:
                discarded_final_fragment = True
                continue
            raise ValueError("GAIA results contain malformed non-final JSON") from exc
        if not isinstance(record, dict):
            raise ValueError("GAIA results record must be a JSON object")
        missing = _RESULT_REQUIRED_KEYS.difference(record)
        if missing:
            raise ValueError(f"GAIA results record is missing schema keys: {sorted(missing)}")
        task_id = record["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("GAIA results record has an invalid task_id")
        if task_id not in allowed_ids:
            raise ValueError(f"GAIA results contain task_id outside manifest: {task_id}")
        if task_id in records:
            raise ValueError(f"GAIA results contain duplicate task_id: {task_id}")
        if not isinstance(record["level"], int) or isinstance(record["level"], bool):
            raise ValueError(f"GAIA results record has an invalid level: {task_id}")
        if not isinstance(record["trace"], list):
            raise ValueError(f"GAIA results record has an invalid trace: {task_id}")
        records[task_id] = record
        valid_lines.append(line)

    if discarded_final_fragment:
        _atomic_write_text(results_path, "\n".join(valid_lines) + ("\n" if valid_lines else ""))
    return records


def run_gaia100(dataset, runtime_factory: Callable[[], AgentRuntime], work_root: str | Path, seed: str = "gaia-local-v2", manifest_path: str | Path | None = None, limit: int | None = None, dataset_revision: str = GAIA_REVISION, run_config: dict | None = None, partition: str = "evaluation") -> dict:
    selected = select_gaia_partition(dataset, partition=partition, seed=seed)
    work_root = Path(work_root)
    results_path = work_root / "results.jsonl"
    manifest = Path(manifest_path) if manifest_path else None
    if results_path.exists() and results_path.stat().st_size > 0 and (manifest is None or not manifest.exists()):
        raise ValueError("GAIA manifest required to resume nonempty results")
    if manifest_path:
        save_local_manifest(selected, manifest_path, seed, dataset_revision, partition=partition)
    _ensure_run_config(work_root, results_path, run_config)
    manifest_task_ids = [str(row["task_id"]) for row in selected]
    existing_results = _load_existing_results(results_path, manifest_task_ids)
    if limit is not None:
        selected = selected[: max(0, limit)]
    work_root.mkdir(parents=True, exist_ok=True)
    results_by_id = dict(existing_results)
    runtime: AgentRuntime | None = None
    for index, row in enumerate(selected, 1):
        task_id = str(row["task_id"])
        if task_id in existing_results:
            continue
        task_dir = work_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        file_path = row.get("file_path")
        if file_path and Path(str(file_path)).exists():
            src = Path(str(file_path))
            shutil.copy2(src, task_dir / src.name)
        question = str(row["Question"])
        if file_path:
            question += f"\nAn attachment is available in the workspace as {Path(str(file_path)).name}."
        question += (
            "\n\nBenchmark answer format: Return only the final answer as a number, "
            "as few words as possible, or a comma-separated list. Do not add explanation."
        )
        if runtime is None:
            runtime = runtime_factory()
        started = time.perf_counter()
        result = runtime.run(question, task_dir)
        latency_ms = (time.perf_counter() - started) * 1000
        correct = gaia_score(result.answer, str(row["Final answer"])) if result.completed else False
        capability_gap = known_capability_gap(file_path)
        failure_label = classify_failure_label(result, correct=correct, capability_gap=capability_gap)
        failure_signals = collect_failure_signals(result, correct=correct, capability_gap=capability_gap)
        record = {
            "index": index,
            "task_id": task_id,
            "level": int(row["Level"]),
            "model_answer": result.answer,
            "correct": correct,
            "completed": result.completed,
            "stop_reason": result.stop_reason,
            "steps": result.metrics.steps,
            "model_calls": result.metrics.steps,
            "latency_ms": latency_ms,
            "tool_calls": result.metrics.tool_calls,
            "tool_successes": result.metrics.tool_successes,
            "tool_errors": result.metrics.tool_errors,
            "duplicate_calls_blocked": result.metrics.duplicate_calls_blocked,
            "capability_gap": capability_gap,
            "failure_label": failure_label,
            "failure_signals": failure_signals,
            "trace": [e.to_dict() for e in result.trace],
        }
        results_by_id[task_id] = record
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    results = [results_by_id[str(row["task_id"])] for row in selected]
    summary = summarize_results(results)
    (work_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
