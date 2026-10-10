from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
            required_checks = ("completed", "final_answer_correct", "zero_tool_errors", "zero_duplicate_blocks", "required_tools_satisfied", "provenance_complete")
            if not isinstance(verification, dict) or any(verification.get(key) is not True for key in required_checks):
                raise ValueError(f"line {line_no}: trajectory policy verification is incomplete")
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


def _render_turn_examples(processor, rows: list[dict]) -> list[dict[str, str]]:
    """Turn verified trajectories into prompt/completion rows.

    Every assistant decision becomes one supervised next-action example. This trains the
    policy at the exact state -> next assistant action boundary while preserving tool
    observations in the prompt history.
    """
    examples: list[dict[str, str]] = []
    for row in rows:
        messages = row["messages"]
        tools = row.get("tools") or []
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
                examples.append({"prompt": prompt, "completion": completion})
    if not examples:
        raise ValueError("No assistant turns available after turnification")
    return examples


def _cpu_safe_compute_loss(model, inputs: dict, *, return_outputs: bool = False):
    """Use the model's native causal-LM loss without TRL's Triton-only fused head."""
    outputs = model(**inputs)
    loss = outputs.loss
    return (loss, outputs) if return_outputs else loss


def _training_runtime_defaults(*, cuda_available: bool, cuda_bf16_supported: bool) -> dict[str, Any]:
    """Return the small set of hardware-dependent training choices.

    CPU uses bfloat16 for 4-bit matmul compute but does not enable Trainer AMP. More
    importantly, CPU disables gradient checkpointing: on the 15 GiB GitHub runner the
    quantized 4B backbone fits, while recomputation made a single optimizer step exceed
    the old 45-minute smoke timeout.
    """
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

    examples = _render_turn_examples(tokenizer, rows)
    dataset = Dataset.from_list(examples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(resume_from_checkpoint).resolve() if resume_from_checkpoint else None
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
        "training_examples": len(examples),
        "policy": "turn-level state -> next assistant action; completion-only loss",
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
        "completion_only_loss": True,
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
