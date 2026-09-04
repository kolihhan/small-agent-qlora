from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from gaia_small_agent.agent.loop import (
    MAX_TOOL_OBSERVATION_CHARS,
    SYSTEM_PROMPT,
    _bound_tool_observation,
)
from gaia_small_agent.agent.types import ToolCall
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, load_validation, select_gaia100
from gaia_small_agent.model.transformers_qwen import TransformersQwenModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.tools.base import ToolResult


def _safe_memory(torch) -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {"allocated": None, "reserved": None, "peak_allocated": None, "free": None}
    free, _total = torch.cuda.mem_get_info()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "free": int(free),
    }


def _reset_peak(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _prompt_tokens(model: TransformersQwenModel, messages: list[dict[str, Any]], tool_defs: list[dict[str, Any]]) -> int | None:
    try:
        encoded = model.processor.apply_chat_template(
            messages,
            tools=tool_defs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        input_ids = encoded.get("input_ids")
        return int(input_ids.shape[-1]) if input_ids is not None else None
    except Exception:
        return None


def _text_tokens(model: TransformersQwenModel, text: str) -> int | None:
    try:
        return len(model.processor.tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def _observation_quality(text: str) -> dict[str, Any]:
    sample = text[:8192]
    nonempty = len(sample)
    controls = sum(1 for char in sample if ord(char) < 32 and char not in "\t\r\n")
    try:
        text.encode("utf-8")
        utf8_valid = True
    except UnicodeEncodeError:
        utf8_valid = False
    return {
        "utf8_valid": utf8_valid,
        "replacement_count": text.count("\ufffd"),
        "nul": "\x00" in sample,
        "control_ratio": controls / nonempty if nonempty else 0.0,
    }


def _classify_exception(exc: BaseException, torch) -> tuple[bool, bool]:
    oom_types = tuple(item for item in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if isinstance(item, type))
    is_oom = bool(oom_types) and isinstance(exc, oom_types)
    if type(exc).__name__ == "RuntimeError":
        text = str(exc).casefold()
        is_oom = is_oom or ("out of memory" in text or "cuda" in text and ("alloc" in text or "memory" in text))
    return is_oom, type(exc).__name__ == "AcceleratorError"


def _load_persisted_ids(results_path: Path) -> set[str]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return set()
    raw = results_path.read_bytes()
    lines = raw.splitlines()
    found: set[str] = set()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith((b"\n", b"\r")):
                continue
            raise
        if isinstance(record, dict) and isinstance(record.get("task_id"), str):
            found.add(record["task_id"])
    return found


def _task_workspace(row: dict[str, Any], workspace: Path) -> None:
    file_path = row.get("file_path")
    if file_path and Path(str(file_path)).exists():
        source = Path(str(file_path))
        shutil.copy2(source, workspace / source.name)


def _emit_event(kind: str, data: dict[str, Any]) -> None:
    print(json.dumps({"event": kind, **data}, ensure_ascii=False, sort_keys=True), file=sys.__stdout__, flush=True)


def _validate_run_config(path: Path) -> None:
    expected = {
        "backend": "transformers",
        "model": "Qwen/Qwen3.5-4B",
        "adapter": None,
        "max_steps": 12,
        "max_new_tokens": 512,
        "thinking": False,
        "quantize_4bit": True,
        "seed": "gaia100-v1",
        "dataset_revision": GAIA_REVISION,
        "tool_observation_max_chars": MAX_TOOL_OBSERVATION_CHARS,
    }
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError


def _run_diagnostic(row: dict[str, Any], workspace: Path, model: TransformersQwenModel) -> dict[str, Any]:
    from gaia_small_agent.agent.loop import _bound_tool_observation

    tools = default_tools()
    tool_defs = [tool.definition() for tool in tools]
    tool_map = {tool.name: tool for tool in tools}
    torch = model.torch
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(row["Question"])},
    ]
    if row.get("file_path"):
        messages[-1]["content"] += f"\nAn attachment is available in the workspace as {Path(str(row['file_path'])).name}."
    messages[-1]["content"] += (
        "\n\nBenchmark answer format: Return only the final answer as a number, "
        "as few words as possible, or a comma-separated list. Do not add explanation."
    )
    seen: set[str] = set()
    terminal_reason = "max_steps"
    terminal_step = 0
    exception_type: str | None = None
    cuda_oom = False

    for step in range(1, 13):
        terminal_step = step
        before_tokens = _prompt_tokens(model, messages, tool_defs)
        before_chars = sum(len(str(message.get("content", ""))) for message in messages)
        _reset_peak(torch)
        memory_before = _safe_memory(torch)
        _emit_event("before_model", {
            "step": step,
            "prompt_chars": before_chars,
            "prompt_tokens": before_tokens,
            "memory": memory_before,
        })
        try:
            turn = model.complete(messages, tool_defs)
        except Exception as exc:
            exception_type = type(exc).__name__
            cuda_oom, accelerator_error = _classify_exception(exc, torch)
            del exc
            _emit_event("after_model", {
                "step": step,
                "prompt_chars": before_chars,
                "prompt_tokens": before_tokens,
                "assistant_output_tokens": None,
                "tool_call_count": None,
                "memory_before": memory_before,
                "memory_after": _safe_memory(torch),
            })
            _emit_event("termination", {"reason": "model_exception", "step": step, "exception_type": exception_type, "cuda_oom": cuda_oom, "accelerator_error": accelerator_error})
            return 1
        after_tokens = _prompt_tokens(model, messages, tool_defs)
        _emit_event("after_model", {
            "step": step,
            "prompt_chars": before_chars,
            "prompt_tokens": before_tokens,
            "prompt_tokens_after": after_tokens,
            "assistant_output_tokens": _text_tokens(model, turn.content or ""),
            "tool_call_count": len(turn.tool_calls),
            "memory_before": memory_before,
            "memory_after": _safe_memory(torch),
        })
        if not turn.tool_calls:
            terminal_reason = "final"
            _emit_event("termination", {"reason": terminal_reason, "step": step, "exception_type": None, "cuda_oom": False, "accelerator_error": False})
            break

        calls = [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}} for call in turn.tool_calls]
        messages.append({"role": "assistant", "content": turn.content or "", "tool_calls": calls})
        for call in turn.tool_calls:
            tool = None
            reported_tool_name = call.name if call.name in tool_map else "unknown"
            key = json.dumps([call.name, call.arguments], sort_keys=True, ensure_ascii=False, default=str)
            if key in seen:
                result = ToolResult(False, "Exact duplicate tool call blocked; choose a different action.", "DUPLICATE_CALL")
            else:
                seen.add(key)
                tool = tool_map.get(call.name)
                if tool is None:
                    result = ToolResult(False, f"Unknown tool: {call.name}", "UNKNOWN_TOOL")
                else:
                    try:
                        result = tool.run(call.arguments, workspace)
                    except Exception as exc:
                        result = ToolResult(False, f"{type(exc).__name__}: {exc}", "TOOL_EXCEPTION")
                        del exc
            raw_observation = result.observation()
            delivered = _bound_tool_observation(raw_observation)
            before_observation_tokens = _prompt_tokens(model, messages, tool_defs)
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": delivered})
            after_observation_tokens = _prompt_tokens(model, messages, tool_defs)
            _emit_event("tool_observation", {
                "step": step,
                "tool_name": reported_tool_name,
                "ok": result.ok,
                "error_code": result.error_code,
                "raw_chars": len(raw_observation),
                "delivered_chars": len(delivered),
                "prompt_tokens_before": before_observation_tokens,
                "prompt_tokens_after": after_observation_tokens,
                "prompt_token_delta": (after_observation_tokens - before_observation_tokens) if before_observation_tokens is not None and after_observation_tokens is not None else None,
                "truncated": delivered != raw_observation,
                **_observation_quality(raw_observation),
            })
    if terminal_reason == "max_steps":
        _emit_event("termination", {"reason": terminal_reason, "step": terminal_step, "exception_type": None, "cuda_oom": False, "accelerator_error": False})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="gaia-benchmark/GAIA")
    parser.add_argument("--manifest", default="runs/gaia100-base/manifest.json")
    parser.add_argument("--results", default="runs/gaia100-base/results.jsonl")
    parser.add_argument("--run-config", default="runs/gaia100-base/run_config.json")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args(argv)

    metadata: dict[str, Any] = {"model_load": {}, "diagnostic": None}
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            _validate_run_config(Path(args.run_config))
            rows = load_validation(args.source, token=args.hf_token, revision=GAIA_REVISION)
            selected = select_gaia100(rows, seed="gaia100-v1")
            manifest_ids = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["task_ids"]
            target_ids = [str(task_id) for task_id in manifest_ids[:48]]
            assert len(target_ids) == 48
            by_id = {str(row["task_id"]): row for row in selected}
            assert all(task_id in by_id for task_id in target_ids)
            persisted = _load_persisted_ids(Path(args.results))
            pending = [by_id[task_id] for task_id in target_ids if task_id not in persisted]
            assert len(pending) == 1
            import torch
            metadata["model_load"] = {
                "cuda_available": bool(torch.cuda.is_available()),
                "attention": "default",
            }
            model = TransformersQwenModel(
                model_name="Qwen/Qwen3.5-4B",
                quantize_4bit=True,
                enable_thinking=False,
                max_new_tokens=512,
            )
            config = getattr(model.model, "config", None)
            text_config = getattr(config, "text_config", None)
            attention = getattr(text_config, "_attn_implementation", None) or getattr(config, "_attn_implementation", None) or "default"
            metadata["model_load"]["attention"] = str(attention)
            metadata["model_load"]["cache_implementation"] = model.cache_implementation
            metadata["model_load"]["memory_after_load"] = _safe_memory(model.torch)
            _emit_event("model_loaded", metadata["model_load"])
            with tempfile.TemporaryDirectory() as temp_dir:
                workspace = Path(temp_dir)
                _task_workspace(pending[0], workspace)
                return _run_diagnostic(pending[0], workspace, model)
    except BaseException as exc:
        metadata["terminal_reason"] = "setup_exception"
        metadata["exception_type"] = type(exc).__name__
        metadata["cuda_oom"] = False
        metadata["accelerator_error"] = False
        del exc
        _emit_event("termination", {"reason": "setup_exception", "step": 0, "exception_type": metadata["exception_type"], "cuda_oom": False, "accelerator_error": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
