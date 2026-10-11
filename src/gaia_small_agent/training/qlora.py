from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


_V3A_CAPABILITIES = (
    "tool_selection",
    "argument_grounding",
    "multi_step",
    "evidence_to_final",
    "no_tool_stop",
    "failure_recovery",
    "strategy_switch",
    "duplicate_avoidance",
)


def load_verified_rows(path: str | Path, protected_questions_path: str | Path | None = None) -> list[dict]:
    from .protection import load_protected_question_hashes, question_hash

    protected_hashes = load_protected_question_hashes(protected_questions_path)
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("verified") is not True:
                raise ValueError(f"line {line_no}: trajectory is not verified")
            provenance = row.get("provenance")
            required_provenance = ("license", "generator", "generator_version", "oracle_type", "oracle_version")
            if not isinstance(provenance, dict) or any(not str(provenance.get(key) or "") for key in required_provenance):
                raise ValueError(f"line {line_no}: incomplete trajectory provenance")
            verification = row.get("verification")
            required_checks = ("completed", "final_answer_correct", "zero_duplicate_blocks", "required_tools_satisfied", "provenance_complete")
            if not isinstance(verification, dict) or any(verification.get(key) is not True for key in required_checks):
                raise ValueError(f"line {line_no}: trajectory policy verification is incomplete")
            zero_tool_errors = verification.get("zero_tool_errors") is True
            expected_failures_only = verification.get("expected_tool_failures_only") is True
            if not zero_tool_errors and not expected_failures_only:
                raise ValueError(f"line {line_no}: trajectory policy verification is incomplete")
            if expected_failures_only:
                error_codes = verification.get("expected_tool_error_codes")
                metrics = row.get("metrics")
                if not isinstance(error_codes, list) or not error_codes:
                    raise ValueError(f"line {line_no}: expected tool failures must declare error codes")
                if not isinstance(metrics, dict) or int(metrics.get("tool_errors", 0)) <= 0:
                    raise ValueError(f"line {line_no}: expected tool failures must record tool_errors")
            if "gaia" in str(row.get("source", "")).casefold():
                raise ValueError(f"line {line_no}: GAIA trajectories are forbidden for training")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"line {line_no}: missing messages")
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = str(message.get("content") or "")
                    if content and question_hash(content) in protected_hashes:
                        raise ValueError(f"line {line_no}: protected GAIA question overlap")
            if not any(m.get("role") == "assistant" for m in messages if isinstance(m, dict)):
                raise ValueError(f"line {line_no}: trajectory has no assistant action")
            rows.append(row)
    if not rows:
        raise ValueError("No verified training rows")
    return rows


def _render_turn_examples(processor, rows: list[dict]) -> list[dict[str, Any]]:
    """Turn verified trajectories into prompt/completion rows plus non-tokenized trace metadata."""
    examples: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, 1):
        messages = row["messages"]
        tools = row.get("tools") or []
        task_id = str(row.get("task_id") or f"row-{row_index}")
        capability = str(row.get("capability") or "unspecified")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            prefix = messages[:index]
            if not prefix:
                continue
            prompt = processor.apply_chat_template(
                prefix,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full = processor.apply_chat_template(
                prefix + [message],
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            if not full.startswith(prompt):
                raise ValueError(
                    "chat template cannot isolate the next assistant action without changing its native format"
                )
            completion = full[len(prompt):]
            if completion.strip():
                action_type = "tool_call" if message.get("tool_calls") else "final"
                examples.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "metadata": {
                            "example_id": f"{task_id}:{index}",
                            "task_id": task_id,
                            "capability": capability,
                            "action_type": action_type,
                            "assistant_message_index": index,
                        },
                    }
                )
    if not examples:
        raise ValueError("No assistant turns available after turnification")
    return examples


def _stable_exposure_key(seed: str, namespace: str, example_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{example_id}".encode("utf-8")).hexdigest()


def _select_balanced_exposure_examples(
    examples: list[dict[str, Any]],
    *,
    count: int = 64,
    seed: str = "qlora-v3a-exposure",
) -> list[dict[str, Any]]:
    """Select the frozen v3A 64-turn exposure set without reusing a source task.

    The seven tool-using capabilities contribute five tool calls and three finals each.
    ``no_tool_stop`` contributes eight finals. This yields 35 tool calls / 29 finals,
    closely matching the full v2 turn corpus while guaranteeing all eight capabilities.
    """
    if count != 64:
        raise ValueError("v3A balanced exposure count must be exactly 64")

    by_capability_action: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in examples:
        metadata = example.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("training example is missing trace metadata")
        capability = str(metadata.get("capability") or "")
        action_type = str(metadata.get("action_type") or "")
        example_id = str(metadata.get("example_id") or "")
        task_id = str(metadata.get("task_id") or "")
        if not capability or action_type not in {"tool_call", "final"} or not example_id or not task_id:
            raise ValueError("training example has incomplete trace metadata")
        by_capability_action[capability][action_type].append(example)

    selected: list[dict[str, Any]] = []
    selected_tasks: set[str] = set()
    for capability in _V3A_CAPABILITIES:
        quotas = {"final": 8} if capability == "no_tool_stop" else {"tool_call": 5, "final": 3}
        for action_type, quota in quotas.items():
            candidates = sorted(
                by_capability_action[capability][action_type],
                key=lambda item: _stable_exposure_key(
                    seed,
                    f"{capability}:{action_type}",
                    str(item["metadata"]["example_id"]),
                ),
            )
            chosen = 0
            for item in candidates:
                task_id = str(item["metadata"]["task_id"])
                if task_id in selected_tasks:
                    continue
                selected.append(item)
                selected_tasks.add(task_id)
                chosen += 1
                if chosen == quota:
                    break
            if chosen != quota:
                raise ValueError(
                    f"not enough unique {capability}/{action_type} examples for quota {quota}: found {chosen}"
                )

    if len(selected) != count:
        raise RuntimeError(f"balanced exposure selected {len(selected)} examples, expected {count}")
    selected.sort(
        key=lambda item: _stable_exposure_key(seed, "training-order", str(item["metadata"]["example_id"]))
    )
    return selected


def _build_training_exposure_report(
    selected: list[dict[str, Any]],
    *,
    seed: str,
    source_example_count: int,
) -> dict[str, Any]:
    metadata = [dict(item["metadata"]) for item in selected]
    example_ids = [str(item["example_id"]) for item in metadata]
    selection_bytes = json.dumps(example_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    capability_counts = Counter(str(item["capability"]) for item in metadata)
    action_type_counts = Counter(str(item["action_type"]) for item in metadata)
    capability_action_counts: dict[str, Counter] = defaultdict(Counter)
    for item in metadata:
        capability_action_counts[str(item["capability"])][str(item["action_type"])] += 1
    return {
        "schema_version": "small-agent-training-exposure/v1",
        "seed": seed,
        "source_example_count": int(source_example_count),
        "selected_example_count": len(selected),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "capability_counts": dict(sorted(capability_counts.items())),
        "action_type_counts": dict(sorted(action_type_counts.items())),
        "capability_action_counts": {
            capability: dict(sorted(counts.items()))
            for capability, counts in sorted(capability_action_counts.items())
        },
        "examples": metadata,
    }


def _tokenize_completion_examples(processor, examples: list[dict[str, Any]], max_length: int) -> list[dict[str, list[int]]]:
    """Keep every supervised completion while left-trimming only old prompt context.

    TRL's normal right truncation can remove the whole completion when the native tool
    schema makes the prompt longer than ``max_length``. Here we tokenize explicitly,
    reserve space for the complete next assistant action, and fill the remaining budget
    with the most recent prompt tokens. Labels mask all prompt tokens, so the objective
    remains completion-only. Trace metadata is intentionally discarded here and never
    enters the model input.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    tokenized: list[dict[str, list[int]]] = []
    for index, example in enumerate(examples, 1):
        prompt_ids = list(processor(example["prompt"], add_special_tokens=False)["input_ids"])
        completion_ids = list(processor(example["completion"], add_special_tokens=False)["input_ids"])
        if not completion_ids:
            raise ValueError(f"example {index}: completion tokenization is empty")
        if len(completion_ids) > max_length:
            raise ValueError(
                f"example {index}: completion exceeds max_length ({len(completion_ids)} > {max_length})"
            )
        prompt_budget = max_length - len(completion_ids)
        kept_prompt = prompt_ids[-prompt_budget:] if prompt_budget else []
        input_ids = kept_prompt + completion_ids
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(kept_prompt) + completion_ids,
            }
        )
    if not tokenized:
        raise ValueError("No tokenized training examples")
    return tokenized


def _cpu_safe_compute_loss(model, inputs: dict, *, return_outputs: bool = False):
    """Use the model's native causal-LM loss without TRL's Triton-only fused head."""
    outputs = model(**inputs)
    loss = outputs.loss
    return (loss, outputs) if return_outputs else loss


def _training_runtime_defaults(*, cuda_available: bool, cuda_bf16_supported: bool) -> dict[str, Any]:
    return {
        "compute_dtype_name": "bfloat16" if (not cuda_available or cuda_bf16_supported) else "float16",
        "use_gradient_checkpointing": bool(cuda_available),
        "trainer_loss_path": "trl-fused" if cuda_available else "native-cpu",
    }


def _normalize_lora_target_modules(value: str | list[str]) -> str | list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "all-linear":
            return stripped
        modules = [item.strip() for item in stripped.split(",") if item.strip()]
    else:
        modules = [str(item).strip() for item in value if str(item).strip()]
    if not modules:
        raise ValueError("lora_target_modules must not be empty")
    return modules


def train_qlora(
    data_path: str | Path,
    output_dir: str | Path,
    model_name: str = "Qwen/Qwen3.5-4B",
    max_length: int = 1536,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    protected_questions_path: str | Path | None = None,
    max_steps: int | None = None,
    save_steps: int = 1,
    gradient_accumulation_steps: int = 8,
    resume_from_checkpoint: str | Path | None = None,
    lora_target_modules: str | list[str] = "all-linear",
    exposure_count: int | None = None,
    exposure_seed: str = "qlora-v3a-exposure",
) -> None:
    rows = load_verified_rows(data_path, protected_questions_path=protected_questions_path)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    cuda_available = torch.cuda.is_available()
    cuda_bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    runtime_defaults = _training_runtime_defaults(
        cuda_available=cuda_available,
        cuda_bf16_supported=cuda_bf16_supported,
    )
    compute_dtype = torch.bfloat16 if runtime_defaults["compute_dtype_name"] == "bfloat16" else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    language_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    language_model = prepare_model_for_kbit_training(
        language_model,
        use_gradient_checkpointing=runtime_defaults["use_gradient_checkpointing"],
    )
    target_modules = _normalize_lora_target_modules(lora_target_modules)
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    language_model = get_peft_model(language_model, peft_cfg)

    for name, param in language_model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            raise RuntimeError(f"Unexpected trainable non-LoRA parameter: {name}")

    all_examples = _render_turn_examples(tokenizer, rows)
    examples = all_examples
    exposure_report = None
    if exposure_count is not None:
        examples = _select_balanced_exposure_examples(
            all_examples,
            count=exposure_count,
            seed=exposure_seed,
        )
        exposure_report = _build_training_exposure_report(
            examples,
            seed=exposure_seed,
            source_example_count=len(all_examples),
        )

    tokenized_examples = _tokenize_completion_examples(tokenizer, examples, max_length=max_length)
    dataset = Dataset.from_list(tokenized_examples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if exposure_report is not None:
        (output_dir / "training-exposure.json").write_text(
            json.dumps(exposure_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    resume_path = Path(resume_from_checkpoint).resolve() if resume_from_checkpoint else None
    prompt_tokens_retained = [sum(1 for label in item["labels"] if label == -100) for item in tokenized_examples]
    completion_tokens = [sum(1 for label in item["labels"] if label != -100) for item in tokenized_examples]
    run_config = {
        "model": model_name,
        "quantization": "4-bit NF4 + double quant",
        "compute_dtype": str(compute_dtype),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": target_modules,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "epochs": epochs,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "resume_from_checkpoint": str(resume_path) if resume_path else None,
        "source_training_examples": len(all_examples),
        "training_examples": len(tokenized_examples),
        "exposure_count": exposure_count,
        "exposure_seed": exposure_seed if exposure_count is not None else None,
        "exposure_selection_sha256": exposure_report["selection_sha256"] if exposure_report else None,
        "policy": "turn-level state -> next assistant action; completion-only loss",
        "truncation_policy": "preserve full completion; left-trim oldest prompt tokens",
        "min_retained_prompt_tokens": min(prompt_tokens_retained),
        "max_retained_prompt_tokens": max(prompt_tokens_retained),
        "min_completion_tokens": min(completion_tokens),
        "max_completion_tokens": max(completion_tokens),
        "scope": "text-only causal language model",
        "trainer_loss_path": runtime_defaults["trainer_loss_path"],
        "gradient_checkpointing": runtime_defaults["use_gradient_checkpointing"],
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": max_length,
        "completion_only_loss": False,
        "loss_type": "nll",
        "logging_steps": 1,
        "save_total_limit": 2,
        "report_to": "none",
        "bf16": bool(cuda_available and compute_dtype == torch.bfloat16),
        "fp16": bool(cuda_available and compute_dtype == torch.float16),
        "seed": 42,
        "data_seed": 42,
    }
    if max_steps is not None:
        config_kwargs.update({
            "max_steps": max_steps,
            "save_strategy": "steps",
            "save_steps": save_steps,
        })
    else:
        config_kwargs["save_strategy"] = "epoch"
    config = SFTConfig(**config_kwargs)

    trainer_cls = SFTTrainer
    if not cuda_available:
        class CpuSafeSFTTrainer(SFTTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                return _cpu_safe_compute_loss(model, inputs, return_outputs=return_outputs)

        trainer_cls = CpuSafeSFTTrainer

    trainer = trainer_cls(
        model=language_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))