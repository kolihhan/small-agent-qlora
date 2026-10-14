from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one resumable CPU-only v3D QLoRA continuation batch")
    parser.add_argument("--batch-index", type=int, required=True, help="Zero-based v3D batch index")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--source-optimizer", type=Path)
    parser.add_argument("--source-state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--policy-seed", default="small-agent-policy-v3")
    parser.add_argument("--v3b-seed", default="qlora-v3b-exposure")
    parser.add_argument("--v3d-seed", default="qlora-v3d-corrective")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from gaia_small_agent.training.exposure_v3d import (
        plan_v3d_batches,
        select_v3d_corrective_examples,
        summarize_v3d_selection,
    )
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    from gaia_small_agent.training.qlora import (
        _render_turn_examples,
        _select_diagnostic_exposure_examples,
        _tokenize_completion_examples,
        load_verified_rows,
    )

    if torch.cuda.is_available():
        raise SystemExit("v3D CPU continuation must not use CUDA")
    if args.batch_index < 0:
        raise SystemExit("batch-index must be non-negative")
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    if not args.source_adapter.is_dir():
        raise SystemExit(f"source adapter directory not found: {args.source_adapter}")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = args.output_dir / "oracle"
    oracle = generate_oracle_policy_v3_dataset(oracle_dir, seed=args.policy_seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = load_verified_rows(oracle["train"])
    all_examples = _render_turn_examples(tokenizer, rows)
    v3b = _select_diagnostic_exposure_examples(all_examples, count=64, seed=args.v3b_seed)
    v3b_task_ids = {str(item["metadata"]["task_id"]) for item in v3b}
    selected = select_v3d_corrective_examples(
        all_examples,
        seed=args.v3d_seed,
        exclude_task_ids=v3b_task_ids,
    )
    selection_report = summarize_v3d_selection(selected)
    batches = plan_v3d_batches(selected, batch_size=args.batch_size)
    if args.batch_index >= len(batches):
        raise SystemExit(f"batch-index {args.batch_index} out of range for {len(batches)} batches")

    prior_examples = [
        item
        for batch in batches[: args.batch_index]
        for item in batch
    ]
    expected_prior_ids = [str(item["metadata"]["example_id"]) for item in prior_examples]
    current = batches[args.batch_index]
    current_ids = [str(item["metadata"]["example_id"]) for item in current]

    prior_state: dict = {}
    if args.source_state is not None:
        if not args.source_state.is_file():
            raise SystemExit(f"source state not found: {args.source_state}")
        prior_state = json.loads(args.source_state.read_text(encoding="utf-8"))
        if prior_state.get("selection_sha256") != selection_report["selection_sha256"]:
            raise SystemExit("source checkpoint selection hash does not match frozen v3D selection")
        if prior_state.get("completed_example_ids") != expected_prior_ids:
            raise SystemExit("source checkpoint completed examples do not match expected batch prefix")
        if int(prior_state.get("optimizer_steps", -1)) != len(expected_prior_ids):
            raise SystemExit("source checkpoint optimizer step count does not match expected prefix")
    elif args.batch_index != 0:
        raise SystemExit("nonzero batch requires source-state for continuity verification")

    if args.batch_index == 0 and (args.source_optimizer is not None or args.source_state is not None):
        raise SystemExit("batch 0 starts a new corrective optimizer from frozen v3B weights")
    if args.batch_index > 0 and args.source_optimizer is None:
        raise SystemExit("resumed batch requires source-optimizer")

    (args.output_dir / "selection.json").write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan_payload = {
        "schema_version": "v3d-cpu-batch-plan/v1",
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "selection_sha256": selection_report["selection_sha256"],
        "expected_prior_example_ids": expected_prior_ids,
        "current_example_ids": current_ids,
    }
    (args.output_dir / "batch-plan.json").write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan_payload, indent=2, sort_keys=True), flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, args.source_adapter, is_trainable=True)

    trainable_params = []
    trainable_names = []
    trainable_parameter_count = 0
    for name, parameter in model.named_parameters():
        keep = "lora_" in name and ".layers.31." in name
        parameter.requires_grad = keep
        if keep:
            trainable_params.append(parameter)
            trainable_names.append(name)
            trainable_parameter_count += parameter.numel()
    if not trainable_params:
        raise SystemExit("no layer-31 LoRA parameters are trainable")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    if args.source_optimizer is not None:
        if not args.source_optimizer.is_file():
            raise SystemExit(f"source optimizer not found: {args.source_optimizer}")
        optimizer.load_state_dict(torch.load(args.source_optimizer, map_location="cpu", weights_only=False))

    model.train()
    timings: list[dict] = []
    completed_ids = list(expected_prior_ids)
    global_step = len(completed_ids)

    for offset, example in enumerate(current, 1):
        example_id = str(example["metadata"]["example_id"])
        tokenized = _tokenize_completion_examples(
            tokenizer,
            [example],
            max_length=args.max_length,
            require_full_prompt=True,
        )[0]
        total_tokens = len(tokenized["input_ids"])
        batch = {
            key: torch.tensor([value], dtype=torch.long)
            for key, value in tokenized.items()
        }

        step_start = time.perf_counter()
        forward_start = time.perf_counter()
        outputs = model(**batch)
        forward_seconds = time.perf_counter() - forward_start
        loss = outputs.loss

        backward_start = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_seconds = time.perf_counter() - optimizer_start
        step_seconds = time.perf_counter() - step_start

        global_step += 1
        completed_ids.append(example_id)
        record = {
            "batch_index": args.batch_index,
            "batch_offset": offset,
            "global_step": global_step,
            "example_id": example_id,
            "task_id": str(example["metadata"]["task_id"]),
            "capability": str(example["metadata"]["capability"]),
            "action_type": str(example["metadata"]["action_type"]),
            "assistant_message_index": int(example["metadata"]["assistant_message_index"]),
            "tokens": total_tokens,
            "loss": float(loss.detach().cpu()),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": optimizer_seconds,
            "step_seconds": step_seconds,
        }
        timings.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        progress = {
            "schema_version": "v3d-cpu-continuation-progress/v1",
            "batch_index": args.batch_index,
            "selection_sha256": selection_report["selection_sha256"],
            "optimizer_steps": global_step,
            "completed_example_ids": completed_ids,
            "latest_step": record,
        }
        (args.output_dir / "progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    checkpoint_dir = args.output_dir / "checkpoint" / "adapter"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    optimizer_path = args.output_dir / "checkpoint" / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)

    state = {
        "schema_version": "v3d-cpu-continuation-state/v1",
        "backend": "github-actions-cpu",
        "cuda_available": torch.cuda.is_available(),
        "model": args.model,
        "initial_adapter": "v3B" if args.batch_index == 0 else "previous-v3D-batch",
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "selection_sha256": selection_report["selection_sha256"],
        "optimizer_steps": global_step,
        "completed_example_ids": completed_ids,
        "current_batch_example_ids": current_ids,
        "trainable_scope": "layer-31 LoRA only",
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_tensor_count": len(trainable_names),
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "batch_step_seconds": sum(item["step_seconds"] for item in timings),
        "timings": timings,
    }
    state_path = args.output_dir / "checkpoint" / "state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "timings.json").write_text(
        json.dumps(timings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    expected_after = len(expected_prior_ids) + len(current)
    if global_step != expected_after or completed_ids != expected_prior_ids + current_ids:
        raise SystemExit("checkpoint continuity verification failed after training")
    print(json.dumps(state, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
