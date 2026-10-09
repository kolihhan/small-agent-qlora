# GAIA Improvement Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the minimum framework needed to develop Small-Agent on Diagnostic25, gate frozen candidates on an unseen blind Shadow25, and preserve Evaluation100 for the final Base / policy / QLoRA comparison.

**Architecture:** Reuse the existing GAIA selector, runner, CLI, verified-trajectory path, and QLoRA trainer. Add only one deterministic `shadow` partition, one aggregate promotion gate, one matched baseline/candidate Shadow workflow, and documentation that makes the candidate lifecycle explicit. Do not add a new agent framework or experiment platform.

**Tech Stack:** Python 3.11, pytest, existing `datasets`/Hugging Face evaluation path, Ollama `qwen3.5:4b` in GitHub Actions, existing GitHub Actions CI.

**Spec:** `docs/superpowers/specs/2026-10-09-gaia-improvement-framework-design.md`

## Global Constraints

- Keep Qwen3.5-4B and the existing `search`, `read`, `inspect`, `python` tool surface unless a later experiment explicitly tests a tool change.
- GAIA revision remains `682dd723ee1e1697e00360edccf2366dc8418dd9`.
- Diagnostic25 remains development-only; `--limit` is debug-only.
- Shadow is blind and aggregate-only; no question, reference answer, model answer, task-level trace, or case-level artifact may be uploaded or printed.
- Evaluation100 stays unopened until final arms are frozen.
- Training data remains non-GAIA, provenance-complete, verified, and protected by the existing leakage guard.
- One candidate = one falsifiable treatment.
- Shadow exposure budget: at most three candidate generations for this improvement cycle.
- No new planner, critic, router, multi-agent system, vector DB, MLflow, experiment DB, or orchestration dependency.

## Review Focus

1. **Partition exhaustion:** pinned GAIA data must fail loudly if the unused rows cannot supply Shadow quotas `8/13/4`; never silently resize the set.
2. **Partition leakage:** Diagnostic25, Shadow25, and Evaluation100 must be pairwise disjoint and deterministic under the pinned revision/seed.
3. **Blind-output leakage:** the Shadow workflow must upload/print only aggregate summaries and protocol metadata, even when a run fails.
4. **False promotion:** completion/tool-efficiency gains without `+2` exact-correct Shadow tasks must not pass the promotion gate.
5. **Runtime failure masking:** a candidate that introduces model/backend/capacity failure must not pass merely because aggregate correctness happens to improve.

---

### Task 1: Add the deterministic Shadow25 partition

**Files:**
- Modify: `src/gaia_small_agent/benchmark/gaia100.py`
- Modify: `tests/test_local_protocol.py`
- Modify: `tests/test_gaia100.py` only if manifest coverage belongs there more naturally

**Interfaces:**
- Consumes: existing `_rank()`, `DIAGNOSTIC_QUOTAS`, `QUOTAS`, `LOCAL_PROTOCOL_SEED`, `select_gaia_partition()`, `save_local_manifest()`.
- Produces: `SHADOW_QUOTAS = {1: 8, 2: 13, 3: 4}` and `select_gaia_partition(..., partition="shadow") -> list[dict]` with deterministic pairwise-disjoint selection.

- [ ] **Step 1: Write failing partition tests**

Add tests asserting:

```python
assert len(diagnostic) == 25
assert len(shadow) == 25
assert len(evaluation) == 100
assert {r["task_id"] for r in diagnostic}.isdisjoint(r["task_id"] for r in shadow)
assert {r["task_id"] for r in diagnostic}.isdisjoint(r["task_id"] for r in evaluation)
assert {r["task_id"] for r in shadow}.isdisjoint(r["task_id"] for r in evaluation)
assert Counter(int(r["Level"]) for r in shadow) == {1: 8, 2: 13, 3: 4}
assert select_gaia_partition(rows, "shadow") == select_gaia_partition(rows, "shadow")
```

Also add a fixture with insufficient unused rows and assert `ValueError` instead of quota shrinkage.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_local_protocol.py tests/test_gaia100.py -q`

Expected: failure because `shadow` is not accepted/implemented.

- [ ] **Step 3: Implement Shadow selection minimally**

Extend `select_gaia_partition()` so each level is deterministically ranked once, Diagnostic rows are reserved first, Evaluation rows are reserved second, and Shadow rows come from the remaining ranked rows using a separate shadow seed namespace. Preserve current Diagnostic25 and Evaluation100 identities; do not reshuffle them.

Extend `save_local_manifest()` so `partition="shadow"` writes Shadow quotas and rejects an existing manifest from another partition.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_local_protocol.py tests/test_gaia100.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add blind GAIA shadow partition`

---

### Task 2: Add an aggregate Shadow promotion gate

**Files:**
- Create: `src/gaia_small_agent/benchmark/promotion.py`
- Modify: `src/gaia_small_agent/benchmark/gaia100.py` or `runner.py` only to expose required aggregate stop-reason counts
- Create: `tests/test_promotion.py`
- Modify: `tests/test_gaia100.py` only if summary-schema coverage belongs there

**Interfaces:**
- Consumes: two aggregate GAIA summary dictionaries produced by existing `summarize_results()`.
- Produces: `evaluate_shadow_promotion(base: dict, candidate: dict) -> dict` with fields `promote`, `correct_delta`, `completion_delta`, `new_catastrophic_failures`, and `reasons`.

- [ ] **Step 1: Write failing promotion tests**

Cover these exact contracts:

```python
# +2 correct, completion not worse by >1, no new catastrophic stop => pass
assert evaluate_shadow_promotion(base, candidate)["promote"] is True

# +1 correct only => fail
# same correct but fewer calls => fail
# +2 correct but completion -2 => fail
# +2 correct but introduces model_capacity/model_timeout/model_unavailable/model_error => fail
```

Add summary coverage for aggregate `stop_reasons` if it does not already exist.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_promotion.py tests/test_gaia100.py -q`

Expected: failure because the promotion helper/required aggregate stop metadata does not exist.

- [ ] **Step 3: Implement the minimum gate**

Define catastrophic stop reasons as: `model_capacity`, `model_timeout`, `model_unavailable`, `model_error`.

Promotion requires all three:

```text
candidate.correct - base.correct >= 2
candidate.completed - base.completed >= -1
no catastrophic stop reason appears in candidate that was absent in baseline
```

Do not use tool calls, latency, duplicate blocks, or tool-success rate as promotion criteria.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_promotion.py tests/test_gaia100.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add aggregate GAIA shadow promotion gate`

---

### Task 3: Expose Shadow through the existing CLI without widening scope

**Files:**
- Modify: `src/gaia_small_agent/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 1 `select_gaia_partition(..., "shadow")` through the existing runner path.
- Produces: `small-agent gaia-eval --partition shadow`.

- [ ] **Step 1: Write the failing CLI contract test**

Assert parser acceptance for `--partition shadow`; keep `diagnostic` and `evaluation` unchanged.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_cli.py -q`

Expected: parser rejects `shadow`.

- [ ] **Step 3: Add `shadow` to the existing partition choices and update help text**

No new command, no new CLI abstraction.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest tests/test_cli.py -q`

- [ ] **Step 5: Commit**

Commit message: `feat: expose blind shadow evaluation`

---

### Task 4: Add one matched, aggregate-only Shadow workflow

**Files:**
- Create: `.github/workflows/gaia-shadow-gate.yml`
- Create: `.github/GAIA_SHADOW_TRIGGER`
- Create or Modify: `tests/test_workflow_contracts.py` (prefer an existing workflow-contract test file if one exists)

**Interfaces:**
- Consumes: `HF_TOKEN`, Ollama `qwen3.5:4b`, a baseline SHA written in `.github/GAIA_SHADOW_TRIGGER`, current candidate SHA, Task 1 Shadow partition, Task 2 promotion helper.
- Produces: one uploaded JSON artifact containing only aggregate baseline summary, aggregate candidate summary, model digest/configuration metadata, candidate/baseline SHAs, exposure number, and promotion verdict.

- [ ] **Step 1: Write a failing static workflow contract test**

Assert the workflow:

- verifies `HF_TOKEN` exists before model pull;
- checks out baseline and candidate refs separately;
- runs both on `--partition shadow` with matched model/step/token/thinking settings;
- never uploads `results.jsonl`, manifests containing task IDs, traces, protected hashes, questions, model answers, or task workspaces;
- uploads only a sanitized aggregate file;
- calls the promotion helper and records the verdict;
- requires an explicit exposure number `1..3` from the trigger metadata.

- [ ] **Step 2: Run the workflow-contract test and verify RED**

Run: `python -m pytest tests/test_workflow_contracts.py -q`

Expected: workflow missing.

- [ ] **Step 3: Implement the workflow**

Use one Ubuntu GitHub-hosted job and one Ollama server/model pull. Run baseline then candidate against the same frozen Shadow protocol. Keep raw GAIA results only in the ephemeral runner workspace and delete them before artifact upload. Generate the final aggregate JSON with the Task 2 gate.

`GAIA_SHADOW_TRIGGER` format stays deliberately small:

```text
baseline_sha=<40-char sha>
exposure=1
candidate_name=<short label>
```

- [ ] **Step 4: Run the static contract test and the complete test suite**

Run:

```text
python -m pytest -q
python -m compileall -q src tests
```

Expected: all tests and compile pass.

- [ ] **Step 5: Commit**

Commit message: `ci: add blind paired GAIA shadow gate`

---

### Task 5: Make the framework the repository's visible experiment contract

**Files:**
- Modify: `README.md`
- Keep: `docs/superpowers/specs/2026-10-09-gaia-improvement-framework-design.md`
- Keep: this plan

**Interfaces:**
- Consumes: implemented Tasks 1–4.
- Produces: one public explanation of DEV25 → Shadow25 → frozen Final100 and the Base / policy / QLoRA research arms.

- [ ] **Step 1: Update README terminology**

State explicitly:

- historical first-13 results are DEV-only and distribution-biased;
- Diagnostic25 is inspectable development evidence;
- Shadow25 is blind with max three exposures in this cycle;
- Evaluation100 is unopened until final arms are frozen;
- completion/runtime gains are not GAIA accuracy gains;
- QLoRA is evidence-gated, not guaranteed.

- [ ] **Step 2: Run doc-adjacent tests and full CI checks**

Run:

```text
python -m pytest -q
python -m compileall -q src tests
```

Expected: clean.

- [ ] **Step 3: Commit**

Commit message: `docs: adopt GAIA improvement framework`

---

### Task 6: Produce the first unseen result without further tuning

**Files:**
- Create candidate branch from the completed framework baseline.
- Modify only `src/gaia_small_agent/agent/loop.py` plus its focused tests for the pre-existing bounded-finalization treatment.
- Update `.github/GAIA_SHADOW_TRIGGER` only to freeze the baseline SHA, candidate label, and exposure `1`.

**Interfaces:**
- Consumes: framework baseline from Tasks 1–5 and the already-developed generic hypothesis: one bounded tool-free finalization recovery for step-limit/empty-final termination.
- Produces: Shadow exposure #1 aggregate verdict. No case-level Shadow inspection.

- [ ] **Step 1: Freeze the hypothesis before candidate code**

Record:

```text
Hypothesis: bounded tool-free finalization recovery reduces incomplete termination without increasing tool use; it counts as GAIA improvement only if blind Shadow exact-correct improves by >=2.
Treatment: one tool-free recovery path covering step-limit and empty-final termination; no planner, no extra tool execution, no prompt content tied to GAIA cases.
```

- [ ] **Step 2: Reapply the treatment on top of the framework baseline with existing RED/GREEN tests**

Use the previously developed recovery behavior, but do not copy unrelated workflow/telemetry changes from older experimental branches.

- [ ] **Step 3: Run full unit/compile verification**

Run:

```text
python -m pytest -q
python -m compileall -q src tests
```

Expected: green before Shadow exposure.

- [ ] **Step 4: Freeze candidate SHA and trigger Shadow exposure #1**

Do not change candidate behavior after the aggregate result is observed.

- [ ] **Step 5: Apply the predeclared verdict**

- If Shadow passes: mark the policy candidate as eligible for the final-arm freeze; do **not** open Evaluation100 yet.
- If Shadow fails: reject this direction. Do not inspect Shadow cases. Return to Diagnostic25/non-GAIA regressions for the next single hypothesis. Exposure budget becomes `2` for the next candidate.

- [ ] **Step 6: Commit only the aggregate experiment record**

Record baseline SHA, candidate SHA, model digest/config, exposure number, aggregate metrics, and `PASS/REJECT`. Never commit gated case content.

Commit message: `eval: record first blind GAIA shadow verdict`

---

## Self-Review Notes

- **Spec coverage:** Runtime, Eval, Diagnose, Improve, Compare, Shadow blindness, exposure budget, and QLoRA isolation are represented. QLoRA curriculum expansion is deliberately deferred until a training hypothesis exists; this matches the spec's evidence-gated rule rather than leaving a gap.
- **No duplicate source of truth:** the framework spec is the design authority; this plan only maps it to files/tests/commands.
- **Treatment isolation:** Task 6 recreates the bounded-finalization candidate cleanly on top of the framework baseline instead of reusing a branch contaminated by experimental workflow changes.
- **Final100 protection:** this plan intentionally stops at the first blind Shadow verdict. Opening Evaluation100 requires freezing all final comparison arms, including the QLoRA arm if QLoRA is justified.
- **Ponytail check:** one new Python helper module, one workflow, no new dependency, no experiment DB, no orchestration package.
