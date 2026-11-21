"""Bounded GitHub-hosted CPU feasibility gate for original Qwen3.5-9B step70→100.

This is deliberately NOT a 9B training run: it checks real host limits first,
before downloading the 19GB+ source weights or risking an OOM / 6h timeout.
A resource PASS would only authorize a later one-step training feasibility trial,
NOT establish exact CUDA FP16 → CPU BF16 parity or GAIA improvement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import urllib.request

MODEL = "Qwen/Qwen3.5-9B"
HF_TREE = "https://huggingface.co/api/models/Qwen/Qwen3.5-9B/tree/main?recursive=true"
GIB = 1024**3
OUTPUT = Path("cpu-v8-feasibility.json")


def cgroup_memory_limit() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        p = Path(path)
        if not p.is_file():
            continue
        try:
            n = int(p.read_text().strip())
        except (ValueError, OSError):
            continue
        if 0 < n < 1 << 50:
            return n
    return None


def memtotal() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux memory total unknown")


def base_weight_bytes() -> tuple[int | None, dict]:
    """Retrieve metadata only, NOT the tensor shards; fail closed if unavailable."""
    try:
        req = urllib.request.Request(HF_TREE, headers={"User-Agent": "small-agent-cpu-resource-gate/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            listing = json.load(resp)
        names = [x for x in listing if x.get("path", "").startswith("model.safetensors-")
                 and x.get("path", "").endswith(".safetensors")]
        if len(names) != 4 or any(type(x.get("size")) is not int or x["size"] <= 0 for x in names):
            return None, {"status": "unknown", "reason": "hf_weight_shards_missing_or_unexpected"}
        return sum(x["size"] for x in names), {
            "status": "verified_public_metadata", "weight_shards": 4,
            "files": [{"name": x["path"], "size_bytes": x["size"]} for x in sorted(names, key=lambda y:y["path"])],
        }
    except Exception as exc:
        return None, {"status": "unknown", "reason": "hf_metadata_not_available", "exception_class": type(exc).__name__}


def assess(*, ram_bytes: int, disk_free_bytes: int, cpu_count: int,
           base_weights_bytes: int | None) -> dict:
    # Conservative LOWER FLOOR, not a guarantee of successful 9B CPU QLoRA.
    # Full HF shards are downloaded before transformers quantizes as it loads.
    # Additional storage is needed for torch/transformers, output checkpoint,
    # intermediate files, and the original verified stage70 source.
    required_disk = base_weights_bytes + 6*GIB if base_weights_bytes is not None else None
    required_ram = 24 * GIB
    reasons = []
    if base_weights_bytes is None:
        reasons.append("original_model_source_weight_bytes_unverified")
    if required_disk is not None and disk_free_bytes < required_disk:
        reasons.append("insufficient_free_disk_to_download_original_model_and_restore_checkpoint")
    if ram_bytes < required_ram:
        reasons.append("insufficient_ram_for_safe_nine_billion_parameter_cpu_training_trial")
    if cpu_count < 4:
        reasons.append("too_few_cpu_cores_for_bounded_qlora_trial")
    # Successful resource gate would not prove optimizer exactness/FP16 parity.
    return {
        "resource_gate": "BLOCKED" if reasons else "RESOURCE_GATE_PASS_ONLY",
        "reasons": reasons,
        "measured": {
            "cpu_count": cpu_count,
            "usable_ram_bytes": ram_bytes,
            "available_disk_bytes": disk_free_bytes,
        },
        "minimum_trial_floor": {
            "cpu_count": 4, "ram_bytes": required_ram,
            "free_disk_bytes": required_disk,
        },
        "source_model_weights_bytes": base_weights_bytes,
        "training_performed": False,
        "original_checkpoint70_unchanged": True,
        "original_t4_precision": "FP16",
        "original_cpu_code_default_precision": "BF16",
        "cross_device_exact_optimizer_continuation_not_proven": True,
        "estimated_cpu_step_duration": "unknown; never claim same T4 throughput",
    }


def main() -> None:
    if not os.environ.get("GITHUB_ACTIONS"):
        print("LOCAL_DIAGNOSTIC_ONLY: this is intended for a real GitHub-hosted runner", flush=True)
    weights, source = base_weight_bytes()
    ram = min(v for v in (memtotal(), cgroup_memory_limit()) if v is not None)
    disk = shutil.disk_usage("/").free
    result = {
        "protocol": "QWEN9B_V8_CPU_STAGE70_TO100_HOST_GATE_V1",
        "repo": "kolihhan/small-agent-qlora",
        "model": MODEL,
        "checkpoint_provenance": {
            "verified_train_step": 70, "target_step": 100,
            "historical_run": 44861054303, "artifact_id": 14693522324,
            "model_name_fixed": True, "no_gaia_training": True,
        },
        "host": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "github_runner_name": os.getenv("RUNNER_NAME", ""),
            "github_runner_os": os.getenv("RUNNER_OS", ""),
        },
        "original_hf_metadata": source,
        "evaluation": assess(ram_bytes=ram, disk_free_bytes=disk,
                             cpu_count=os.cpu_count() or 0, base_weights_bytes=weights),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("QWEN9B_CPU_RESOURCE_GATE", json.dumps({
        "status": result["evaluation"]["resource_gate"],
        "reasons": result["evaluation"]["reasons"],
        "weights_gib": round(weights/GIB, 2) if weights else None,
        "usable_ram_gib": round(ram/GIB, 2),
        "free_disk_gib": round(disk/GIB, 2),
        "cpu_cores": os.cpu_count() or 0,
        "training_performed": False,
    },sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
