from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
import gaia_small_agent.benchmark.runner as gaia_runner
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import default_tools


def _ids_hash(rows: list[dict]) -> str:
    payload = "\n".join(str(row["task_id"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _aggregate_trace_counts(results_path: Path) -> dict:
    error_codes = Counter()
    tool_calls_by_name = Counter()
    tool_successes_by_name = Counter()
    tool_errors_by_name = Counter()
    error_codes_by_tool: dict[str, Counter] = defaultdict(Counter)

    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for event in record.get("trace", []):
            kind = event.get("kind")
            data = event.get("data") or {}
            if kind == "tool_call":
                name = str(data.get("name") or "unknown")
                tool_calls_by_name[name] += 1
            elif kind == "tool_result":
                name = str(data.get("name") or "unknown")
                if bool(data.get("ok")):
                    tool_successes_by_name[name] += 1
                else:
                    tool_errors_by_name[name] += 1
                    code = str(data.get("error_code") or "UNKNOWN")
                    error_codes[code] += 1
                    error_codes_by_tool[name][code] += 1

    return {
        "error_codes": dict(sorted(error_codes.items())),
        "tool_calls_by_name": dict(sorted(tool_calls_by_name.items())),
        "tool_successes_by_name": dict(sorted(tool_successes_by_name.items())),
        "tool_errors_by_name": dict(sorted(tool_errors_by_name.items())),
        "error_codes_by_tool": {
            name: dict(sorted(codes.items()))
            for name, codes in sorted(error_codes_by_tool.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = load_validation("gaia-benchmark/GAIA", token=args.hf_token, revision=GAIA_REVISION)
    full_selection = select_gaia_partition(dataset, partition="diagnostic", seed=LOCAL_PROTOCOL_SEED)
    shard = shard_selected_rows(full_selection, shard_index=args.shard_index, shard_count=args.shard_count)
    if not shard:
        raise SystemExit("selected GAIA diagnostic shard is empty")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    tool_names = [tool.name for tool in default_tools()]

    def factory() -> AgentRuntime:
        model = OllamaModel(
            model=args.model,
            base_url=args.ollama_url,
            max_new_tokens=512,
            enable_thinking=False,
        )
        return AgentRuntime(model, default_tools(), max_steps=12)

    run_config = {
        "backend": "ollama",
        "model": args.model,
        "adapter": None,
        "max_steps": 12,
        "max_new_tokens": 512,
        "thinking": False,
        "quantize_4bit": None,
        "seed": LOCAL_PROTOCOL_SEED,
        "dataset_revision": GAIA_REVISION,
        "tool_observation_max_chars": MAX_TOOL_OBSERVATION_CHARS,
        "partition": "diagnostic",
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
            partition="diagnostic",
        )
    finally:
        gaia_runner.select_gaia_partition = original_selector

    if int(summary["total"]) != len(shard):
        raise SystemExit(f"diagnostic shard summary total mismatch: {summary['total']} != {len(shard)}")

    diagnostics = _aggregate_trace_counts(work_root / "results.jsonl")
    payload = {
        "schema_version": "gaia-9b-error-diagnostic/v1",
        "benchmark": "GAIA 2023 validation / frozen Diagnostic25",
        "dataset_revision": GAIA_REVISION,
        "seed": LOCAL_PROTOCOL_SEED,
        "model": args.model,
        "thinking": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_selection_sha256": _ids_hash(full_selection),
        "shard_selection_sha256": _ids_hash(shard),
        "summary": summary,
        "error_codes": diagnostics["error_codes"],
        "tool_calls_by_name": diagnostics["tool_calls_by_name"],
        "tool_successes_by_name": diagnostics["tool_successes_by_name"],
        "tool_errors_by_name": diagnostics["tool_errors_by_name"],
        "error_codes_by_tool": diagnostics["error_codes_by_tool"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
