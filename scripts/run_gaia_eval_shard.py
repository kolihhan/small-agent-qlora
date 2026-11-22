from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.agent.leakage_gate import LeakageGate
from gaia_small_agent.protocol_recovery import ProtocolRepairModel
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
import gaia_small_agent.benchmark.runner as gaia_runner
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import check_default_tools, default_tools
from gaia_small_agent.training.protection import build_protected_question_hashes


def _ids_hash(rows: list[dict]) -> str:
    payload = "\n".join(str(row["task_id"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "openai_compat"], default="ollama")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--benchmark-leakage-gate", action="store_true", help="research only; gold-blind suppress public benchmark solutions")
    parser.add_argument("--protocol-json-retry", action="store_true", help="research only; retry malformed native tool arguments at most twice per task")
    parser.add_argument("--final-text-retry", action="store_true", help="research only; one extra final-only response if last step emits textual tool call")
    parser.add_argument("--partition", choices=("evaluation", "shadow"), default="evaluation", help="research-only frozen partition; never change existing evaluation default")
    parser.add_argument("--evidence-stop-contract", action="store_true", help="research-only self-decision evidence stop/continue instruction; does not force final")
    parser.add_argument("--max-tool-observation-chars", type=int, choices=(2000, 8000), default=MAX_TOOL_OBSERVATION_CHARS, help="research-only matched observation budget; default unchanged")
    return parser


def build_model(args):
    if args.backend == "openai_compat":
        return OpenAICompatModel(
            model=args.model,
            base_url=args.openai_base_url,
            max_new_tokens=512,
        )
    return OllamaModel(
        model=args.model,
        base_url=args.ollama_url,
        max_new_tokens=512,
        enable_thinking=args.thinking,
    )


def preflight_eval_tools(workspace: str | Path) -> dict:
    report = check_default_tools(workspace, live_search=False)
    if not report["ready"]:
        failed = [
            f"{name}: {check.get('error_code') or check.get('detail') or 'failed'}"
            for name, check in report["checks"].items()
            if not check.get("ok")
        ]
        raise RuntimeError("evaluation tool preflight failed: " + "; ".join(failed))
    return {
        "tool_names": report["tool_names"],
        "dependencies": report["dependencies"],
        "smokes": report["smokes"],
    }


def main() -> int:
    args = build_parser().parse_args()

    with tempfile.TemporaryDirectory(prefix="small-agent-shard-preflight-") as workspace:
        tool_preflight = preflight_eval_tools(workspace)

    dataset = load_validation("gaia-benchmark/GAIA", token=args.hf_token, revision=GAIA_REVISION)
    full_selection = select_gaia_partition(dataset, partition=args.partition, seed=LOCAL_PROTOCOL_SEED)
    shard = shard_selected_rows(full_selection, shard_index=args.shard_index, shard_count=args.shard_count)
    if not shard:
        raise SystemExit("selected GAIA shard is empty")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    build_protected_question_hashes(dataset, work_root / "protected-question-hashes.json")

    tool_names = tool_preflight["tool_names"]

    def factory() -> AgentRuntime:
        model = build_model(args)
        if args.protocol_json_retry:
            model = ProtocolRepairModel(model, max_retries=2)
        return AgentRuntime(model, default_tools(), max_steps=12,
                            observation_filter=LeakageGate() if args.benchmark_leakage_gate else None,
                            final_text_retry=args.final_text_retry,
                            evidence_stop_contract=args.evidence_stop_contract,
                            max_tool_observation_chars=args.max_tool_observation_chars)

    run_config = {
        "backend": args.backend,
        "model": args.model,
        "adapter": None,
        "max_steps": 12,
        "max_new_tokens": 512,
        "thinking": args.thinking,
        "benchmark_leakage_gate": bool(args.benchmark_leakage_gate),
        "protocol_json_retry": bool(args.protocol_json_retry),
        "final_text_retry": bool(args.final_text_retry),
        "evidence_stop_contract": bool(args.evidence_stop_contract),
        "max_additional_protocol_calls_per_task": 2 if args.protocol_json_retry else 0,
        "max_additional_final_calls_per_task": 1 if args.final_text_retry else 0,
        "quantize_4bit": None,
        "seed": LOCAL_PROTOCOL_SEED,
        "dataset_revision": GAIA_REVISION,
        "tool_observation_max_chars": args.max_tool_observation_chars,
        "partition": args.partition,
        "cache_implementation": None,
        "tool_names": tool_names,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_selection_sha256": _ids_hash(full_selection),
        "shard_selection_sha256": _ids_hash(shard),
    }

    original_selector = gaia_runner.select_gaia_partition
    gaia_runner.select_gaia_partition = lambda rows, partition, seed: list(shard)
    try:
        summary = gaia_runner.run_gaia100(
            dataset,
            factory,
            work_root,
            seed=LOCAL_PROTOCOL_SEED,
            manifest_path=None,
            dataset_revision=GAIA_REVISION,
            run_config=run_config,
            partition=args.partition,
        )
    finally:
        gaia_runner.select_gaia_partition = original_selector

    if int(summary["total"]) != len(shard):
        raise SystemExit(f"shard summary total mismatch: {summary['total']} != {len(shard)}")

    payload = {
        "schema_version": "gaia-9b-shard/v1",
        "benchmark": ("GAIA 2023 validation / reserved Shadow25" if args.partition == "shadow" else "GAIA 2023 validation / frozen Evaluation100"),
        "dataset_revision": GAIA_REVISION,
        "seed": LOCAL_PROTOCOL_SEED,
        "partition": args.partition,
        "backend": args.backend,
        "model": args.model,
        "thinking": args.thinking,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_selection_sha256": _ids_hash(full_selection),
        "shard_selection_sha256": _ids_hash(shard),
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
