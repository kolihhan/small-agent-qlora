# Production Runtime Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the existing local Small-Agent runtime at real operational boundaries without changing the single-model architecture or the QLoRA experiment conclusion.

**Architecture:** Keep `AgentRuntime` and the existing tool surface. Add a small typed model-failure contract, bounded source reads, an operator-facing `doctor` command, self-describing run artifacts, and cross-platform CI. Avoid new services, orchestration layers, or training/evaluation changes.

**Tech Stack:** Python 3.11, argparse, requests, pytest, GitHub Actions, existing local Qwen/Ollama/Transformers adapters.

**Spec:** `docs/superpowers/specs/2026-10-09-production-runtime-hardening-design.md`

## Global Constraints

- Do not change benchmark/result numbers or QLoRA conclusions.
- Do not add planner/router/critic architecture, hosted services, containers, databases, or retry frameworks.
- Unexpected programming bugs outside known model/tool boundaries must remain visible exceptions.
- Keep the ordinary runtime usable without installing the training stack.
- CI must remain CPU-only and must not download/run Qwen or GAIA.

## Review Focus

- Ollama timeout, connection refusal, HTTP failure, and malformed JSON must stop cleanly without reporting completion.
- An oversized HTTP body must be rejected while streaming rather than fully materialized in memory.
- An oversized workspace file must be rejected before PDF/XLSX/text parsing.
- `doctor` must aggregate failures instead of aborting after the first failed check.
- Existing `small-agent run --output` consumers must still find `answer`, `completed`, `stop_reason`, `metrics`, and `trace` at the top level.

---

### Task 1: Structured model runtime failures

**Files:**
- Modify: `src/gaia_small_agent/agent/types.py`
- Modify: `src/gaia_small_agent/agent/loop.py`
- Modify: `src/gaia_small_agent/model/ollama.py`
- Modify: `src/gaia_small_agent/model/transformers_qwen.py`
- Modify: `tests/test_agent_loop.py`
- Modify: `tests/test_ollama.py`
- Modify: `tests/test_qwen_protocol.py`

**Interfaces:**
- Produces: `ModelRuntimeError(stop_reason: str, message: str)` with stable stop reasons `model_timeout`, `model_unavailable`, or `model_error`.
- Consumes: existing `AgentRuntime.run()` and `ModelCapacityError` behavior.

- [ ] **Step 1: Write failing agent-loop tests**

Add tests asserting `ModelRuntimeError("model_timeout", "...")` and `ModelRuntimeError("model_unavailable", "...")` produce incomplete results with matching `stop_reason` and a final trace event containing the message.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_agent_loop.py`
Expected: FAIL because `ModelRuntimeError` handling does not exist.

- [ ] **Step 3: Implement the typed model failure contract and loop handling**

Add `ModelRuntimeError` in `agent/types.py`; catch it in `AgentRuntime.run()` after `ModelCapacityError`, record one `model_error` trace event, and return an incomplete result using its stable stop reason.

- [ ] **Step 4: Add failing Ollama mapping tests**

Test Requests timeout → `model_timeout`, connection error → `model_unavailable`, generic request/HTTP failure → `model_error`, and malformed/missing `message` payload → `model_error`.

- [ ] **Step 5: Implement minimal Ollama mappings**

Catch only known `requests` boundary failures and response-shape failures; rethrow as `ModelRuntimeError` without retrying.

- [ ] **Step 6: Add failing Transformers inference test**

Test a non-capacity generation exception is converted to `ModelRuntimeError("model_error", ...)` while capacity behavior remains unchanged.

- [ ] **Step 7: Implement Transformers inference mapping**

Keep `_is_capacity_error` precedence; map other exceptions raised inside `complete()` to `model_error`.

- [ ] **Step 8: Verify Task 1**

Run: `python -m pytest -q tests/test_agent_loop.py tests/test_ollama.py tests/test_qwen_protocol.py`
Expected: PASS.

- [ ] **Step 9: Run full suite**

Run: `python -m pytest -q`
Expected: PASS.

### Task 2: Bound source reads before expensive parsing

**Files:**
- Create: `src/gaia_small_agent/tools/limits.py`
- Modify: `src/gaia_small_agent/tools/read.py`
- Modify: `src/gaia_small_agent/tools/inspect.py`
- Modify: `tests/test_read_tool.py`
- Modify: `tests/test_tool_preflight.py`

**Interfaces:**
- Produces: `MAX_REMOTE_BYTES` and `MAX_WORKSPACE_FILE_BYTES` constants.
- Produces: `ReadTool` error code `CONTENT_TOO_LARGE` for over-budget HTTP/local sources.
- Produces: `InspectTool` error code `CONTENT_TOO_LARGE` for over-budget files.

- [ ] **Step 1: Write failing oversized-local-file tests**

Create sparse/temporary files just above `MAX_WORKSPACE_FILE_BYTES`; assert `read` and `inspect` reject them before parser-specific work with `CONTENT_TOO_LARGE`.

- [ ] **Step 2: Write failing streamed-HTTP-limit test**

Use a fake streamed response whose chunks exceed `MAX_REMOTE_BYTES`; assert the reader stops and returns `CONTENT_TOO_LARGE` without calling `.text`/fully materializing the body.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_read_tool.py tests/test_tool_preflight.py`
Expected: FAIL because source-byte limits do not exist.

- [ ] **Step 4: Implement source-byte limits**

Add limits module. Change HTTP GET to `stream=True`, read bounded chunks, and close responses. Check local file size before reading/parsing. Apply the same local limit in `InspectTool`.

- [ ] **Step 5: Bound PDF/XLSX extraction by requested output**

Stop accumulating PDF pages/XLSX rows once the collected text reaches the requested `max_chars`; retain final slicing as a second bound.

- [ ] **Step 6: Verify Task 2**

Run: `python -m pytest -q tests/test_read_tool.py tests/test_tool_preflight.py`
Expected: PASS.

- [ ] **Step 7: Run full suite**

Run: `python -m pytest -q`
Expected: PASS.

### Task 3: Operator doctor, CLI validation, and run provenance

**Files:**
- Create: `src/gaia_small_agent/doctor.py`
- Modify: `src/gaia_small_agent/cli.py`
- Modify: `src/gaia_small_agent/tools/__init__.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_doctor.py`

**Interfaces:**
- Produces: `run_doctor(backend: str, model: str, ollama_url: str, workspace: Path) -> dict` with top-level `ready: bool` and per-check objects.
- Produces: `small-agent doctor` exit `0` when ready, `2` otherwise.
- Produces: `small-agent-run/v1` output metadata while preserving existing result keys at top level.

- [ ] **Step 1: Write failing CLI numeric validation tests**

Assert `--max-steps 0`, negative steps, zero/negative `--max-new-tokens` are argparse errors.

- [ ] **Step 2: Write failing run-artifact provenance test**

Update the existing output test to require top-level existing result keys plus `schema_version == "small-agent-run/v1"` and a `run_config` containing backend/model/max steps/max tokens/thinking/tool names/tool observation limit.

- [ ] **Step 3: Write failing doctor tests**

Cover: all checks ready; Ollama unreachable; requested model absent; one local tool smoke failing while remaining checks are still reported; Transformers dependency check without model loading.

- [ ] **Step 4: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_cli.py tests/test_doctor.py`
Expected: FAIL because doctor/provenance/positive-number validation do not exist.

- [ ] **Step 5: Implement positive integer argparse type and run metadata**

Use one helper for positive integer arguments. Extend atomic `write_run_output()` to merge `schema_version`, `run_config`, and existing `AgentResult.to_dict()` fields.

- [ ] **Step 6: Implement non-destructive local tool readiness helper**

Refactor the existing preflight so doctor can smoke `read`, `inspect`, and `python` locally and check the search dependency without requiring a live web query. Keep the sealed GAIA preflight's live-search behavior unchanged.

- [ ] **Step 7: Implement `doctor.py` and CLI command**

For Ollama query `/api/tags` with a short timeout and inspect model names. For Transformers import required runtime packages only. Aggregate all check results into one JSON report.

- [ ] **Step 8: Verify Task 3**

Run: `python -m pytest -q tests/test_cli.py tests/test_doctor.py tests/test_tool_preflight.py`
Expected: PASS.

- [ ] **Step 9: Run full suite**

Run: `python -m pytest -q`
Expected: PASS.

### Task 4: Cross-platform CI and production-facing documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/verification.md`

**Interfaces:**
- Consumes: `small-agent doctor`, `small-agent-run/v1`, current install extras.
- Produces: Linux + Windows CPU CI and documented runtime/experiment split.

- [ ] **Step 1: Update CI matrix**

Use `matrix.os: [ubuntu-latest, windows-latest]`, keep Python 3.11, current install/test/compile commands, and add `small-agent --help` after install.

- [ ] **Step 2: Update README**

Add a short production-runtime section: `pip install -e ".[search,files]"`, `small-agent doctor`, one run example with `--output`, what the artifact records, and separate evaluation/training extras. Preserve all experiment numbers and caveats.

- [ ] **Step 3: Update verification doc**

State that current CI verifies Linux and Windows runtime/package behavior and remains separate from model/benchmark evidence. Do not hard-code a test count.

- [ ] **Step 4: Verify locally testable repository state**

Run: `python -m pytest -q && python -m compileall -q src tests`
Expected: PASS.

- [ ] **Step 5: Verify GitHub Actions on both matrix jobs**

Expected: Ubuntu PASS and Windows PASS for the branch head.
