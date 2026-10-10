from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from .benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation
from .benchmark.runner import run_gaia100
from .benchmark.compare import compare_runs
from .doctor import run_doctor
from .model.ollama import OllamaModel
from .model.openai_compat import OpenAICompatModel
from .model.transformers_qwen import TransformersQwenModel
from .tools import default_tools, preflight_default_tools
from .training.qlora import train_qlora
from .training.trajectories import collect_verified_trajectories
from .training.protection import build_protected_question_hashes
from .training.policy_tasks import generate_policy_tasks
from .training.oracle_policy import generate_oracle_policy_dataset
from .training.policy_eval import evaluate_policy_tasks
from .training.policy_promotion import evaluate_policy_promotion


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def make_runtime(args) -> AgentRuntime:
    if args.backend == "transformers":
        model = TransformersQwenModel(
            model_name=args.hf_model,
            quantize_4bit=not args.no_4bit,
            adapter=args.adapter,
            enable_thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
            cache_implementation=None if args.cache_implementation == "default" else args.cache_implementation,
        )
    elif args.backend == "openai_compat":
        if args.adapter:
            raise ValueError("--adapter is loaded by the external OpenAI-compatible server; do not pass it to small-agent")
        model = OpenAICompatModel(
            model=args.model,
            base_url=args.openai_base_url,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        if args.adapter:
            raise ValueError("--adapter requires --backend transformers; Ollama adapters must be exported separately")
        model = OllamaModel(
            model=args.model,
            base_url=args.ollama_url,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.thinking,
        )
    return AgentRuntime(model, default_tools(), max_steps=args.max_steps)


def print_trace(result) -> None:
    for event in result.trace:
        if event.kind == "tool_call":
            print(f"[{event.step}] -> {event.data['name']}({json.dumps(event.data['arguments'], ensure_ascii=False)})")
        elif event.kind == "tool_result":
            status = "ok" if event.data["ok"] else f"error:{event.data.get('error_code')}"
            preview = str(event.data.get("content", "")).replace("\n", " ")[:180]
            print(f"[{event.step}] <- {event.data['name']} [{status}] {preview}")
        elif event.kind == "final":
            print(f"[{event.step}] => final")


def _model_name(args) -> str:
    return args.hf_model if args.backend == "transformers" else args.model


def _single_run_config(args, runtime: AgentRuntime) -> dict:
    return {
        "backend": args.backend,
        "model": _model_name(args),
        "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "thinking": args.thinking,
        "tool_observation_max_chars": MAX_TOOL_OBSERVATION_CHARS,
        "tool_names": list(runtime.tools),
    }


def write_run_output(result, output_path: str | Path, run_config: dict | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["schema_version"] = "small-agent-run/v1"
    payload["run_config"] = dict(run_config or {})
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, ensure_ascii=False)
            temp_file.flush()
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def run_command(args) -> int:
    runtime = make_runtime(args)
    result = runtime.run(args.question, args.workspace)
    print_trace(result)
    print("\nAnswer:\n" + (result.answer or f"<stopped: {result.stop_reason}>"))
    if args.output is not None:
        write_run_output(result, args.output, _single_run_config(args, runtime))
    return 0 if result.completed else 2


def doctor_command(args) -> int:
    model = args.hf_model if args.backend == "transformers" else args.model
    base_url = args.openai_base_url if args.backend == "openai_compat" else args.ollama_url
    report = run_doctor(args.backend, model, base_url, args.workspace)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


def gaia_command(args) -> int:
    with tempfile.TemporaryDirectory(prefix="small-agent-preflight-") as workspace:
        tool_preflight = preflight_default_tools(workspace)
    dataset = load_validation(args.source, token=args.hf_token, revision=args.gaia_revision)
    build_protected_question_hashes(dataset, args.protected_questions)

    def factory():
        return make_runtime(args)

    run_config = {
        "backend": args.backend,
        "model": _model_name(args),
        "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "thinking": args.thinking,
        "quantize_4bit": (not args.no_4bit) if args.backend == "transformers" else None,
        "seed": args.seed,
        "dataset_revision": args.gaia_revision,
        "tool_observation_max_chars": MAX_TOOL_OBSERVATION_CHARS,
        "partition": args.partition,
        "cache_implementation": args.cache_implementation if args.backend == "transformers" else None,
        "tool_names": tool_preflight["tool_names"],
    }
    manifest = args.manifest or str(Path(args.work_root) / "manifest.json")
    summary = run_gaia100(
        dataset,
        factory,
        args.work_root,
        seed=args.seed,
        manifest_path=manifest,
        limit=args.limit,
        dataset_revision=args.gaia_revision,
        run_config=run_config,
        partition=args.partition,
    )
    print(json.dumps(summary, indent=2))
    return 0


def collect_command(args) -> int:
    runtime = make_runtime(args)
    summary = collect_verified_trajectories(
        args.tasks,
        args.output,
        runtime_factory=lambda: runtime,
        work_root=args.work_root,
        protected_questions_path=args.protected_questions,
    )
    print(json.dumps(summary, indent=2))
    return 0


def train_command(args) -> int:
    train_qlora(
        args.data,
        args.output,
        model_name=args.hf_model,
        max_length=args.max_length,
        epochs=args.epochs,
        protected_questions_path=args.protected_questions,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        lora_target_modules=args.lora_target_modules,
    )
    return 0


def generate_policy_tasks_command(args) -> int:
    path = generate_policy_tasks(args.output, count=args.count, seed=args.seed)
    print(json.dumps({"output": str(path), "count": args.count, "seed": args.seed}, indent=2))
    return 0


def generate_oracle_policy_command(args) -> int:
    paths = generate_oracle_policy_dataset(args.output_dir, seed=args.seed)
    payload = {"seed": args.seed, "paths": {name: str(path) for name, path in paths.items()}}
    print(json.dumps(payload, indent=2))
    return 0


def eval_policy_command(args) -> int:
    runtime = make_runtime(args)
    summary = evaluate_policy_tasks(args.tasks, lambda: runtime, args.work_root)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


def compare_policy_command(args) -> int:
    baseline = json.loads(Path(args.base).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    verdict = evaluate_policy_promotion(baseline, candidate, min_exact_pp=args.min_exact_pp)
    text = json.dumps(verdict, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0 if verdict["promote"] else 2


def compare_command(args) -> int:
    result = compare_runs(args.base, args.adapter_run)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


def add_runtime_args(p):
    p.add_argument("--backend", choices=["ollama", "transformers", "openai_compat"], default="ollama")
    p.add_argument("--model", default="qwen3.5:4b", help="Ollama/OpenAI-compatible model name")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--openai-base-url", default="http://127.0.0.1:8080", help="OpenAI-compatible server base URL")
    p.add_argument("--hf-model", default="Qwen/Qwen3.5-4B", help="Transformers model id/path")
    p.add_argument("--adapter", default=None, help="LoRA adapter path; Transformers backend only")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading for Transformers backend")
    p.add_argument("--thinking", action="store_true", help="Enable Qwen thinking mode (Ollama/Transformers only)")
    p.add_argument("--max-new-tokens", type=_positive_int, default=512)
    p.add_argument("--max-steps", type=_positive_int, default=12)
    p.add_argument("--cache-implementation", choices=["offloaded", "default"], default="default", help="Transformers KV-cache placement")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="small-agent")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one autonomous agent task")
    run.add_argument("question")
    run.add_argument("--workspace", default="runs/task")
    run.add_argument("--output", default=None, help="Persist the complete run result as JSON")
    add_runtime_args(run)
    run.set_defaults(func=run_command)

    doctor = sub.add_parser("doctor", help="Check local runtime readiness without loading model weights")
    doctor.add_argument("--backend", choices=["ollama", "transformers", "openai_compat"], default="ollama")
    doctor.add_argument("--model", default="qwen3.5:4b", help="Ollama/OpenAI-compatible model name")
    doctor.add_argument("--ollama-url", default="http://localhost:11434")
    doctor.add_argument("--openai-base-url", default="http://127.0.0.1:8080")
    doctor.add_argument("--hf-model", default="Qwen/Qwen3.5-4B", help="Transformers model id/path")
    doctor.add_argument("--workspace", default="runs/doctor")
    doctor.set_defaults(func=doctor_command)

    gaia = sub.add_parser("gaia-eval", aliases=["gaia100"], help="Run a GAIA diagnostic, blind shadow, or frozen evaluation partition")
    gaia.add_argument("--source", default="gaia-benchmark/GAIA", help="HF dataset repo or local snapshot path")
    gaia.add_argument("--hf-token", default=None)
    gaia.add_argument("--partition", choices=["diagnostic", "shadow", "evaluation"], default="diagnostic")
    gaia.add_argument("--seed", default=LOCAL_PROTOCOL_SEED)
    gaia.add_argument("--gaia-revision", default=GAIA_REVISION, help="Pinned GAIA dataset revision")
    gaia.add_argument("--manifest", default=None, help="Defaults to <work-root>/manifest.json")
    gaia.add_argument("--protected-questions", default="runs/gaia-protected-question-hashes.json")
    gaia.add_argument("--work-root", default="runs/gaia-diagnostic")
    gaia.add_argument("--limit", type=int, default=None, help="Debug only; omit for the full selected partition")
    add_runtime_args(gaia)
    gaia.set_defaults(func=gaia_command)

    collect = sub.add_parser("collect-trajectories", help="Run non-GAIA oracle tasks and save only verified tool-use trajectories")
    collect.add_argument("--tasks", required=True, help="JSONL with id/source/question/expected_answer")
    collect.add_argument("--output", required=True, help="Verified trajectory JSONL")
    collect.add_argument("--work-root", default="runs/trajectory-collection")
    collect.add_argument("--protected-questions", default="runs/gaia-protected-question-hashes.json")
    add_runtime_args(collect)
    collect.set_defaults(func=collect_command)

    train = sub.add_parser("train-qlora", help="Train a LoRA adapter from verified non-GAIA trajectories")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--hf-model", default="Qwen/Qwen3.5-4B")
    train.add_argument("--max-length", type=int, default=1536)
    train.add_argument("--epochs", type=float, default=2.0)
    train.add_argument("--max-steps", type=_positive_int, default=None, help="Absolute trainer step target; use with checkpoints for CPU batching")
    train.add_argument("--save-steps", type=_positive_int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=_positive_int, default=8)
    train.add_argument("--resume-from-checkpoint", default=None)
    train.add_argument("--lora-target-modules", default="all-linear", help="Comma-separated PEFT target module names or all-linear")
    train.add_argument("--protected-questions", default="runs/gaia-protected-question-hashes.json")
    train.set_defaults(func=train_command)

    policy = sub.add_parser("generate-policy-tasks", help="Generate the legacy deterministic policy curriculum")
    policy.add_argument("--output", required=True)
    policy.add_argument("--count", type=int, default=64)
    policy.add_argument("--seed", default="p4-policy-v1")
    policy.set_defaults(func=generate_policy_tasks_command)

    oracle = sub.add_parser("generate-oracle-policy", help="Generate the TinyAgent-style deterministic oracle policy dataset")
    oracle.add_argument("--output-dir", required=True)
    oracle.add_argument("--seed", default="tinyagent-policy-v1")
    oracle.set_defaults(func=generate_oracle_policy_command)

    policy_eval = sub.add_parser("eval-policy", help="Evaluate Base or QLoRA on deterministic held-out tool-policy tasks")
    policy_eval.add_argument("--tasks", required=True)
    policy_eval.add_argument("--work-root", default="runs/policy-eval")
    policy_eval.add_argument("--output", default=None)
    add_runtime_args(policy_eval)
    policy_eval.set_defaults(func=eval_policy_command)

    policy_compare = sub.add_parser("compare-policy-evals", help="Apply the preregistered held-out QLoRA promotion gate")
    policy_compare.add_argument("--base", required=True)
    policy_compare.add_argument("--candidate", required=True)
    policy_compare.add_argument("--min-exact-pp", type=float, default=10.0)
    policy_compare.add_argument("--output", default=None)
    policy_compare.set_defaults(func=compare_policy_command)

    compare = sub.add_parser("compare-evals", help="Compare frozen Base and +LoRA local evaluation runs")
    compare.add_argument("--base", required=True)
    compare.add_argument("--adapter-run", required=True)
    compare.add_argument("--output", default=None)
    compare.set_defaults(func=compare_command)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
