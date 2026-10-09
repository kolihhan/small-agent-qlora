# TinyAgent-style QLoRA Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic TinyAgent-style oracle tool-policy curriculum, evaluate Base vs QLoRA on a held-out split, and gate any GAIA Shadow use on a +10pp held-out improvement.

**Architecture:** Keep the existing Small Agent runtime and QLoRA trainer. Add a deterministic oracle trajectory generator that emits the same verified trajectory schema already consumed by `load_verified_rows`, plus a small generic policy evaluator. QLoRA hyperparameters remain unchanged so curriculum is the only training variable.

**Tech Stack:** Python 3.11, pytest, existing Qwen/OpenAI tool-call message schema, Transformers/TRL/PEFT/bitsandbytes for training.

**Spec:** `docs/superpowers/specs/2026-10-09-tinyagent-qlora-policy-design.md`

## Global Constraints

- QLoRA target is `conversation state + four tool schemas -> next assistant action`, not knowledge injection.
- Dataset is exactly 2,000 rows: 1,600 train / 200 dev / 200 test, balanced across five capabilities.
- No GAIA content or GAIA-derived trajectories in training data.
- Keep existing QLoRA hyperparameters unchanged.
- No runtime, planner, ToolRAG, multi-agent, vector DB, or tool-surface changes.
- Held-out promotion requires at least +10 percentage points exact success before another Shadow exposure.

## Review Focus

- Split leakage: train/dev/test rows must have disjoint task IDs and deterministic independent seed namespaces.
- Tool distractors: all four real tool schemas must be present in every generated row.
- Native tool-call shape: generated assistant tool calls must match existing trajectory/Qwen format.
- No-tool tasks: `no_tool_stop` rows must contain no tool call and still satisfy training verification.
- Multi-step rows: tool observations must precede the next assistant decision and no hidden reasoning may be stored.

---

### Task 1: Oracle curriculum generator

**Files:**
- Create: `src/gaia_small_agent/training/oracle_policy.py`
- Create: `tests/test_oracle_policy.py`

**Interfaces:**
- Produces: `generate_oracle_policy_dataset(output_dir: str | Path, *, seed: str = "tinyagent-policy-v1") -> dict[str, Path]`
- Output files: `train.jsonl`, `dev.jsonl`, `test.jsonl` containing verified trajectory rows accepted by `load_verified_rows`.

- [ ] Write failing tests asserting exact 1600/200/200 counts, five-way balance, deterministic output, disjoint IDs, non-GAIA provenance, four tool schemas on every row, and valid tool/no-tool message shapes.
- [ ] Run `python -m pytest tests/test_oracle_policy.py -q` and confirm failures are due to missing module/function.
- [ ] Implement the minimal deterministic generator with five capability families and independent split RNG namespaces.
- [ ] Run `python -m pytest tests/test_oracle_policy.py -q` and then `python -m pytest -q`; both must pass.
- [ ] Commit as `feat: add TinyAgent-style oracle policy curriculum`.

### Task 2: Training-contract compatibility

**Files:**
- Modify: `tests/test_training_contract.py`
- Modify only if required by tests: `src/gaia_small_agent/training/qlora.py`

**Interfaces:**
- Consumes: generated oracle rows from Task 1.
- Produces: proof that `load_verified_rows` and `_render_turn_examples` accept oracle tool-call and no-tool rows without weakening existing GAIA/provenance guards.

- [ ] Add failing tests loading representative generated rows and rendering tool-call/no-tool assistant turns with a deterministic fake tokenizer contract.
- [ ] Run focused tests and confirm the expected incompatibility if any.
- [ ] Make the smallest compatibility fix only if the current trainer cannot consume the native generated rows.
- [ ] Run focused tests plus full pytest suite.
- [ ] Commit as `test: lock oracle trajectories to QLoRA contract` or `fix: accept oracle policy trajectories` depending on whether production code changes.

### Task 3: Held-out policy evaluator

**Files:**
- Create: `src/gaia_small_agent/training/policy_eval.py`
- Create: `tests/test_policy_eval.py`

**Interfaces:**
- Produces: `evaluate_policy_tasks(tasks_path, runtime_factory, work_root) -> dict` with total, correct, completed, exact_success_rate, required_tools_satisfied, tool_calls, tool_errors, duplicate_calls_blocked, stop_reasons.
- Uses task definitions embedded in the generated dataset metadata or a companion deterministic task manifest produced by Task 1.

- [ ] Write failing tests for exact scoring, required-tool satisfaction, no-tool tasks, aggregate metrics, and catastrophic stop-reason accounting.
- [ ] Run focused tests to confirm RED.
- [ ] Implement the smallest evaluator using existing `gaia_score`/AgentResult metrics rather than a second scoring system.
- [ ] Run focused and full suites.
- [ ] Commit as `feat: add held-out tool-policy evaluator`.

### Task 4: Promotion gate and CLI surface

**Files:**
- Create: `src/gaia_small_agent/training/policy_promotion.py`
- Modify: `src/gaia_small_agent/cli.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_policy_promotion.py`

**Interfaces:**
- Produces: `evaluate_policy_promotion(baseline: dict, candidate: dict, min_exact_pp: float = 10.0) -> dict`.
- CLI exposes dataset generation and policy evaluation without altering existing commands.

- [ ] Write RED tests for +10pp pass, +9.9pp reject, and new catastrophic-failure reject.
- [ ] Write RED CLI contract tests for deterministic dataset generation/evaluation entry points.
- [ ] Implement minimal gate and CLI wiring.
- [ ] Run focused tests and full suite.
- [ ] Commit as `feat: gate QLoRA on held-out policy gain`.

### Task 5: End-to-end reproducibility and documentation

**Files:**
- Modify: `README.md`
- Create: `docs/experiments/tinyagent-qlora-v1.md`

**Interfaces:**
- Records exact branch/base commit, dataset seed/counts, unchanged hyperparameters, held-out gate, and commands for Base / QLoRA evaluation.

- [ ] Add a static/reproducibility test only if README/CLI claims introduce a machine-checkable contract.
- [ ] Generate the 2,000-row curriculum and record hashes/counts.
- [ ] Run full CI on Ubuntu and Windows.
- [ ] Record actual Base held-out result when compute is available, train QLoRA, record candidate held-out result, and apply the +10pp gate.
- [ ] Only if the adapter passes, freeze it for the existing Shadow protocol; otherwise stop v1 without consuming Shadow.
