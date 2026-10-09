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


def _training_precision(torch) -> tuple[Any, bool, bool]:
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


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
    compute_dtype, use_bf16, use_fp16 = _training_precision(torch)
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
    language_model = prepare_model_for_kbit_training(language_model)
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    language_model = get_peft_model(language_model, peft_cfg)

    # Hard boundary: the trainer receives only the language backbone, so LoRA cannot
    # be injected into the vision encoder or other multimodal parent modules.
    for name, param in language_model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            raise RuntimeError(f"Unexpected trainable non-LoRA parameter: {name}")

    examples = _render_turn_examples(tokenizer, rows)
    dataset = Dataset.from_list(examples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": model_name,
        "quantization": "4-bit NF4 + double quant",
        "compute_dtype": str(compute_dtype),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "epochs": epochs,
        "training_examples": len(examples),
        "policy": "turn-level state -> next assistant action; completion-only loss",
        "scope": "text-only causal language model",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_length=max_length,
        completion_only_loss=True,
        loss_type="nll",
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
    )
    trainer = SFTTrainer(
        model=language_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
