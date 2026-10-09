from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from ..agent.loop import AgentRuntime
from ..model.ollama import OllamaModel
from ..tools import default_tools
from .scoring import gaia_score


SCHEMA_VERSION = "small-agent-heldout/v1"
_ALLOWED_CATEGORIES = {"read", "inspect", "compute", "multi_step"}
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_cases(path: str | Path) -> list[dict]:
    path = Path(path)
    cases: list[dict] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid benchmark JSON on line {line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: benchmark case must be an object")
        case_id = str(row.get("id") or "")
        if not _SAFE_ID.fullmatch(case_id):
            raise ValueError(f"line {line_no}: invalid case id")
        if case_id in seen:
            raise ValueError(f"duplicate benchmark case id: {case_id}")
        seen.add(case_id)
        category = str(row.get("category") or "")
        if category not in _ALLOWED_CATEGORIES:
            raise ValueError(f"line {line_no}: invalid category: {category}")
        if row.get("source") != "local-heldout-v1" or row.get("license") != "CC0-1.0":
            raise ValueError(f"line {line_no}: missing frozen held-out provenance")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"line {line_no}: question must be non-empty")
        if not isinstance(row.get("expected_answer"), str) or not row["expected_answer"].strip():
            raise ValueError(f"line {line_no}: expected_answer must be non-empty")
        required_tools = row.get("required_tools")
        if not isinstance(required_tools, list) or not required_tools or any(not isinstance(name, str) or not name for name in required_tools):
            raise ValueError(f"line {line_no}: required_tools must be a non-empty list")
        files = row.get("files") or {}
        if not isinstance(files, dict) or any(not isinstance(name, str) or not isinstance(content, str) for name, content in files.items()):
            raise ValueError(f"line {line_no}: files must map string paths to string content")
        cases.append(row)
    if not cases:
        raise ValueError("benchmark contains no cases")
    return cases


def _materialize_files(case: dict, workspace: Path) -> None:
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for raw_name, content in (case.get("files") or {}).items():
        target = (root / raw_name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"case {case['id']}: file path escapes workspace") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _used_tools(result) -> list[str]:
    return [
        str(event.data.get("name"))
        for event in result.trace
        if event.kind == "tool_call" and event.data.get("name")
    ]


def _summary(cases: list[dict]) -> dict:
    total = len(cases)
    correct = sum(bool(row["correct"]) for row in cases)
    completed = sum(bool(row["completed"]) for row in cases)
    tool_pass = sum(bool(row["required_tools_satisfied"]) for row in cases)
    full_pass = sum(bool(row["full_pass"]) for row in cases)
    tool_calls = sum(int(row["metrics"]["tool_calls"]) for row in cases)
    tool_successes = sum(int(row["metrics"]["tool_successes"]) for row in cases)
    by_category: dict[str, dict] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in cases:
        groups[str(row["category"])].append(row)
    for category in sorted(groups):
        rows = groups[category]
        n = len(rows)
        category_correct = sum(bool(row["correct"]) for row in rows)
        category_full = sum(bool(row["full_pass"]) for row in rows)
        by_category[category] = {
            "total": n,
            "correct": category_correct,
            "accuracy": category_correct / n,
            "full_pass": category_full,
            "full_pass_rate": category_full / n,
        }
    return {
        "accuracy": correct / total,
        "correct": correct,
        "completion_rate": completed / total,
        "completed": completed,
        "required_tool_pass_rate": tool_pass / total,
        "required_tool_pass": tool_pass,
        "full_pass_rate": full_pass / total,
        "full_pass": full_pass,
        "tool_calls": tool_calls,
        "tool_successes": tool_successes,
        "tool_errors": sum(int(row["metrics"]["tool_errors"]) for row in cases),
        "duplicate_calls_blocked": sum(int(row["metrics"]["duplicate_calls_blocked"]) for row in cases),
        "tool_success_rate": tool_successes / tool_calls if tool_calls else 1.0,
        "average_steps": sum(int(row["metrics"]["steps"]) for row in cases) / total,
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in cases) / total,
        "by_category": by_category,
    }


def run_heldout(
    cases: list[dict],
    *,
    runtime_factory: Callable[[], AgentRuntime],
    work_root: str | Path,
    model: dict,
) -> dict:
    started_at = _utc_now()
    work_root = Path(work_root)
    rows: list[dict] = []
    for case in cases:
        workspace = work_root / str(case["id"])
        if workspace.exists():
            shutil.rmtree(workspace)
        _materialize_files(case, workspace)
        runtime = runtime_factory()
        start = time.perf_counter()
        result = runtime.run(str(case["question"]), workspace)
        latency_ms = (time.perf_counter() - start) * 1000.0
        used_tools = _used_tools(result)
        required_tools = [str(name) for name in case["required_tools"]]
        required_ok = set(required_tools).issubset(set(used_tools))
        correct = bool(result.completed and gaia_score(result.answer, str(case["expected_answer"])))
        full_pass = bool(result.completed and correct and required_ok)
        rows.append({
            "id": str(case["id"]),
            "category": str(case["category"]),
            "question": str(case["question"]),
            "expected_answer": str(case["expected_answer"]),
            "answer": result.answer,
            "completed": bool(result.completed),
            "stop_reason": result.stop_reason,
            "correct": correct,
            "required_tools": required_tools,
            "used_tools": used_tools,
            "required_tools_satisfied": required_ok,
            "full_pass": full_pass,
            "latency_ms": latency_ms,
            "metrics": result.to_dict()["metrics"],
            "trace": result.to_dict()["trace"],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "benchmark": "local-heldout-v1",
        "sample_size": len(rows),
        "model": dict(model),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "metrics": _summary(rows),
        "cases": rows,
    }


def _ollama_model_identity(model: str, base_url: str) -> dict:
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Ollama /api/tags response is missing models")
    for row in rows:
        if not isinstance(row, dict):
            continue
        names = {str(row.get("name") or ""), str(row.get("model") or "")}
        if model in names:
            return {"name": model, "digest": str(row.get("digest") or "")}
    raise RuntimeError(f"Ollama model not found: {model}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen local held-out tool-use benchmark")
    parser.add_argument("--cases", default="evaluation/local-heldout-v1/cases.jsonl")
    parser.add_argument("--output", default="runs/local-heldout-v1/report.json")
    parser.add_argument("--work-root", default="runs/local-heldout-v1/work")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.max_new_tokens <= 0 or args.timeout <= 0:
        parser.error("runtime limits must be positive")

    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    model_identity = _ollama_model_identity(args.model, args.ollama_url)

    def runtime_factory() -> AgentRuntime:
        return AgentRuntime(
            OllamaModel(
                model=args.model,
                base_url=args.ollama_url,
                timeout_s=args.timeout,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=False,
            ),
            default_tools(),
            max_steps=args.max_steps,
        )

    report = run_heldout(cases, runtime_factory=runtime_factory, work_root=args.work_root, model=model_identity)
    report["benchmark_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    report["runtime"] = {
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "thinking": False,
        "ollama_url": args.ollama_url,
    }
    report["execution"] = {
        "git_sha": os.getenv("GITHUB_SHA"),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "runner_os": os.getenv("RUNNER_OS"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(output)

    compact = {
        "status": report["status"],
        "benchmark": report["benchmark"],
        "sample_size": report["sample_size"],
        "model": report["model"],
        "benchmark_sha256": report["benchmark_sha256"],
        "metrics": report["metrics"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    for row in report["cases"]:
        print(
            f"CASE {row['id']} category={row['category']} correct={row['correct']} "
            f"tools={row['used_tools']} required_ok={row['required_tools_satisfied']} "
            f"full_pass={row['full_pass']} answer={row['answer']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
