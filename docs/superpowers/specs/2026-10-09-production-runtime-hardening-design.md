# Production Runtime Hardening Design

## Goal

Make the existing local Small-Agent runtime feel production-ready without changing the core single-model agent architecture or manufacturing a QLoRA success claim.

## Scope

This change hardens four operational boundaries:

1. Model/backend failures become explicit structured stop reasons instead of uncaught tracebacks during an agent run.
2. Tool I/O is bounded before expensive reads/parsing can consume unbounded memory.
3. Operators get a lightweight `small-agent doctor` readiness check and reproducible single-run output metadata.
4. CI verifies the supported local runtime on both Linux and Windows; README separates ordinary runtime usage from the heavier experiment/training stack.

## Non-goals

- No planner/router/critic graph.
- No FastAPI service, Docker/Kubernetes, Redis, database, or hosted deployment layer.
- No new QLoRA curriculum, training run, benchmark claim, or evaluation tuning.
- No claim that the Python tool or URL reader is a hostile multi-tenant sandbox.
- No broad retry framework. Failures remain visible and deterministic.

## Runtime error contract

Introduce a model runtime error type carrying a stable stop reason and a human-readable message.

The agent loop must handle:

- `model_capacity` for existing CUDA/resource exhaustion behavior.
- `model_timeout` for backend request timeouts.
- `model_unavailable` for inability to reach the configured local backend.
- `model_error` for other backend/model execution failures that are known at the model boundary.

A handled model failure produces a trace event and an incomplete `AgentResult`; it must not be converted into a successful final answer. Unexpected programming errors outside the model boundary remain exceptions.

The Ollama adapter maps Requests timeout/connection/request failures and malformed responses into the model runtime contract. The Transformers adapter preserves the existing capacity classification and maps other inference-time exceptions into `model_error`.

## Bounded I/O contract

Tool outputs are already capped before re-entering the model, but reads must also be bounded before memory-heavy processing.

- Public HTTP reads use streamed responses and enforce a maximum downloaded-byte budget before decoding.
- Workspace files have a maximum source-file byte budget before PDF/XLSX/text parsing.
- PDF extraction stops once enough text has been collected for the requested output cap.
- XLSX reading stops once enough text has been collected for the requested output cap.
- File inspection rejects oversized files before expensive inspection.
- Limits are module constants with tests pinning their behavior; they are not a general security sandbox.

## Doctor command

Add `small-agent doctor` as a fast readiness check.

Default behavior follows the selected backend and must not load the full Transformers model.

For the default Ollama backend it checks:

- the configured Ollama endpoint is reachable;
- the requested model appears in `/api/tags`;
- the default local tool surface is present;
- local `read`, `inspect`, and `python` smoke checks work;
- optional search support is installed; live web search itself is not required for doctor success.

The command prints a JSON report and exits `0` when ready, `2` otherwise. It should report individual checks rather than abort at the first failure.

For the Transformers backend, doctor checks that the runtime/training dependency imports required by the current implementation are available, but it does not load model weights.

## Single-run artifact provenance

`small-agent run --output ...` keeps the existing result fields and adds:

- `schema_version: "small-agent-run/v1"`
- `run_config` with backend/model identity, adapter when relevant, `max_steps`, `max_new_tokens`, thinking flag, tool observation cap, and tool names.

The write remains atomic. Console output remains concise and unchanged in spirit.

## CLI validation

Runtime numeric limits exposed on the CLI must reject non-positive `max_steps` and `max_new_tokens` at argument parsing time rather than passing invalid values deeper into the runtime.

## CI and packaging verification

Current CI remains lightweight and does not run models or benchmarks.

- Test on `ubuntu-latest` and `windows-latest` using Python 3.11.
- Install `.[dev,files]`.
- Run `python -m pytest -q` and `python -m compileall -q src tests`.
- Add a `small-agent --help` packaging/entry-point smoke check.

No GPU/model CI is added.

## Documentation

README must distinguish:

- normal local runtime install: core + search/files extras;
- evaluation extras;
- QLoRA/Transformers training extras;
- `doctor` before first run;
- single-run JSON provenance;
- existing experiment evidence and limitations remain unchanged.

## Success criteria

- Existing behavior remains green.
- New model failure paths fail closed into stable stop reasons.
- Oversized remote/local content is rejected before unbounded reads.
- `small-agent doctor` returns a structured report and stable exit status.
- Single-run artifacts are self-describing enough to reproduce the runtime configuration.
- Linux and Windows CI pass.
- No benchmark/result numbers or QLoRA conclusions are changed.
