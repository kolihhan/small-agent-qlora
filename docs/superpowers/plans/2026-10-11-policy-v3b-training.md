# Diagnostic-driven QLoRA v3B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v3A smoke curriculum with a minimal, diagnostic-driven 9B QLoRA probe that teaches observation-conditioned decisions without changing the agent runtime or LoRA hyperparameters.

**Architecture:** Keep the existing four-tool AgentRuntime, Qwen3.5-9B, completion-only objective, GAIA anti-leakage boundary, and frozen Evaluation100. Add a v3 synthetic policy generator with structurally distinct train/dev/test prompts, a weighted 64-turn v3B exposure selector based on Diagnostic25 failure evidence, and a fail-closed full-context token audit before any GPU training.

**Tech Stack:** Python 3.11, pytest, Transformers/Qwen chat template, PEFT/QLoRA, GitHub Actions, Hugging Face ZeroGPU.

**Spec:** Diagnostic25 artifact from run 38147196289 plus current branch `qlora/gaia-9b-v2`.

## Global Constraints

- GAIA remains evaluation-only; no GAIA question, answer, trace, entity, or case-specific information enters training.
- Do not change AgentRuntime, tool implementations, Qwen3.5-9B, LoRA r=16, alpha=32, dropout=0.05, LR=1e-4, or all-linear target modules in this iteration.
- Preserve completion-only supervision and the native Qwen tool chat template.
- v3B is a learning probe, not a promotion result; frozen Evaluation100 is consumed only after synthetic held-out gates pass.
- Fail closed if the selected v3B examples cannot retain their full prompt and completion in the configured training context.

## Review Focus

- Recovery examples must require inference from tool observations rather than user instructions such as “if search fails, switch”.
- Tool-selection examples must not name the required tool in the user request.
- Train/dev/test must use different prompt structures, not only different random IDs.
- v3B exposure must emphasize evidence-to-final and multi-step decisions while retaining all eight policy capabilities.
- Full-context audit must catch any truncation before a GPU reservation is consumed.

---

### Task 1: Policy v3 dataset

**Files:**
- Create: `src/gaia_small_agent/training/oracle_policy_v3.py`
- Create: `tests/test_oracle_policy_v3.py`

- [ ] Write tests for split counts, capability balance, structural split differences, and no explicit recovery/tool-name policy leakage.
- [ ] Run CI and observe the tests fail because v3 does not exist.
- [ ] Implement the minimal deterministic v3 generator using existing trajectory helpers.
- [ ] Run the full test suite.

### Task 2: Diagnostic-weighted exposure

**Files:**
- Modify: `src/gaia_small_agent/training/qlora.py`
- Create: `tests/test_training_exposure_v3b.py`

- [ ] Write a failing test for the frozen 64-turn v3B capability/action quotas.
- [ ] Implement `_select_diagnostic_exposure_examples` without altering the v3A selector.
- [ ] Add an explicit `exposure_profile` switch to `train_qlora` and record it in `run_config.json`.
- [ ] Run the full test suite.

### Task 3: Full-context fail-closed audit

**Files:**
- Modify: `src/gaia_small_agent/training/qlora.py`
- Modify: `tests/test_qlora_completion_preserving_tokens.py`
- Create: `.github/workflows/qlora-v3b-token-length-audit.yml`

- [ ] Write a failing test proving strict mode rejects prompt truncation.
- [ ] Add opt-in strict full-context tokenization while preserving legacy behavior for old experiments.
- [ ] Add a Qwen3.5-9B v3B token audit workflow that reports 512/768/1024/1280/1536/2048 fit rates and the minimum safe context.
- [ ] Run CI, then run the audit before creating or launching the final ZeroGPU v3B training workflow.
