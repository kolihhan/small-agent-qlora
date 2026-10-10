from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
import gaia_small_agent.benchmark.runner as gaia_runner
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.protection import build_protected_question_hashes


def _ids_hash(rows: list[dict]) -> str:
    payload = "\n".join(str(row["task_id"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    dataset = load_validation("gaia-benchmark/GAIA", token=args.hf_token, revision=GAIA_REVISION)
    full_selection = select_gaia_partition(dataset, partition="evaluation", seed=LOCAL_PROTOCOL_SEED)
    shard = shard_selected_rows(full_selection, shard_index=args.shard_index, shard_count=args.shard_count)
    if not shard:
        raise SystemExit("selected GAIA shard is empty")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    build_protected_question_hashes(dataset, work_root / "protected-question-hashes.json")

    tool_names = [tool.name for tool in default_tools()]

    def factory() -> AgentRuntime:
        model = OllamaModel(
            model=args.model,
            base_url=args.ollama_url,
            max_new_tokens=512,
            enable_thinking=args.thinking,
        )
        return AgentRuntime(model, default_tools(), max_steps=12)

    run_config = {
        "backend": "ollama",
        "model": args.model,
        "adapter": None,
        "max_steps": 12,
        "max_new_tokens": 512,
        "thinking": args.thinking,
        "quantize_4bit": None,
        "seed": LOCAL_PROTOCOL_SEED,
        "dataset_revision": GAIA_REVISION,
        "tool_observation_max_chars": MAX_TOOL_OBSERVATION_CHARS,
        "partition": "evaluation",
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
            partition="evaluation",
        )
    finally:
        gaia_runner.select_gaia_partition = original_selector

    if int(summary["total"]) != len(shard):
        raise SystemExit(f"shard summary total mismatch: {summary['total']} != {len(shard)}")

    payload = {
        "schema_version": "gaia-9b-shard/v1",
        "benchmark": "GAIA 2023 validation / frozen Evaluation100",
        "dataset_revision": GAIA_REVISION,
        "seed": LOCAL_PROTOCOL_SEED,
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
