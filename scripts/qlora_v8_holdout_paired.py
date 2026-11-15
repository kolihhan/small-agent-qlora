"""One-shot same-Qwen base-v8 matched teacher-forced held-out evaluation.

Usage after a physically verified full 100-step v8 adapter exists:
python -m scripts.qlora_v8_holdout_paired --holdout qlora-v8-holdout.jsonl \
    --audit qlora-v8-audit.json --adapter-dir /path/to/extracted-adapter \
    --output eval-v8-holdout.json

Not a GAIA benchmark and NOT sufficient to promote the v8 adapter.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import time

from gaia_small_agent.training.v8_holdout import paired_v8_actions, select_v8_targets

MODEL = "Qwen/Qwen3.5-9B"
MAX_NEW_TOKENS = 512


def validate_inputs(holdout: Path, audit: Path, adapter: Path) -> dict:
    if not holdout.is_file() or not audit.is_file():
        raise FileNotFoundError("missing frozen v8 held-out rows or audit")
    manifest = json.loads(audit.read_text(encoding="utf-8"))
    if (manifest.get("train_model") != MODEL
        or manifest.get("train_parents") != 304
        or manifest.get("train_actions") != 798
        or manifest.get("validation_parents") != 35
        or manifest.get("validation_actions") != 176
        or manifest.get("no_gaia_training") is not True
        or manifest.get("gaia_evaluated") is not False):
        raise ValueError("v8 source audit contract drift")
    if sha256(holdout.read_bytes()).hexdigest() != manifest.get("validation_sha256"):
        raise ValueError("v8 held-out source SHA256 drift")
    report_file = adapter / "v8-verified-report.json"
    weight_file = adapter / "adapter_model.safetensors"
    config_file = adapter / "adapter_config.json"
    if not all(f.is_file() for f in (report_file, weight_file, config_file)):
        raise FileNotFoundError("no physically verified v8 adapter")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (report.get("model") != MODEL or report.get("optimizer_steps") != 100
        or report.get("train_actions") != 798
        or report.get("train_sha256") != manifest.get("train_sha256")
        or report.get("holdout_parents_unseen") != 35
        or report.get("gaia_evaluated") is not False
        or report.get("adapter_initialized") != "from-base-not-v7"
        or not isinstance(report.get("source_sha"), str)
        or len(report["source_sha"]) != 40):
        raise ValueError("v8 checkpoint/source identity mismatch")
    actual_sha = sha256(weight_file.read_bytes()).hexdigest()
    if actual_sha != report.get("checkpoint_sha256"):
        raise ValueError("v8 safetensors digest mismatch")
    rows = [json.loads(line) for line in holdout.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_v8_targets(rows)
    return {"audit": manifest, "checkpoint": report, "selected": selected}


def evaluate(holdout: Path, audit: Path, adapter: Path) -> dict:
    verified = validate_inputs(holdout, audit, adapter)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("v8 9B paired inference requires actual CUDA, not CPU smoke tests")
    if torch.cuda.get_device_properties(0).total_memory < 14 * 2**30:
        raise RuntimeError("insufficient GPU memory for the same 9B base-v8 comparison")
    from gaia_small_agent.model.transformers_qwen import TransformersQwenModel
    from gaia_small_agent.training.next_action_eval import evaluate_next_actions

    arms = {}
    times = {}
    selected = set(verified["selected"])
    for name, adapter_path in (("base", None), ("v8", str(adapter))):
        started = time.monotonic()
        model = TransformersQwenModel(
            model_name=MODEL, quantize_4bit=True, adapter=adapter_path,
            enable_thinking=False, max_new_tokens=MAX_NEW_TOKENS,
            load_compute_dtype=True,
        )
        # IMPORTANT: the factory returns the existing model, not a fresh
        # nine-billion-parameter load for every one of the 70 target actions.
        arms[name] = evaluate_next_actions(holdout, lambda: model, example_ids=selected)
        times[name] = round(time.monotonic() - started, 1)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    score = paired_v8_actions(verified["selected"], arms["base"], arms["v8"])
    return {
        "protocol": "qwen9b-v8-35-parent-teacher-forced-v1",
        "model": MODEL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "holdout_sha256": verified["audit"]["validation_sha256"],
        "train_sha256": verified["audit"]["train_sha256"],
        "checkpoint_sha256": verified["checkpoint"]["checkpoint_sha256"],
        "training_source_sha": verified["checkpoint"]["source_sha"],
        "latency_s": times,
        "score": score,
        "promotion_allowed": False,
        "limits": "Teacher-forced last tool call and final only; cannot establish autonomous GAIA, multi-hop or stop/completion improvement",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwriting prior independent evaluation")
    report = evaluate(args.holdout, args.audit, args.adapter_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("QLORA_V8_HELDOUT_MATCHED", json.dumps({
        k: v for k, v in report["score"].items() if k != "cases"
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
