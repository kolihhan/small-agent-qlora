<h1 align="center">Small-Agent QLoRA</h1>

<p align="center">
  <strong>Can a local 4B tool agent learn a better tool-use policy with QLoRA — without hiding behind a larger model or benchmark leakage?</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-111827?style=flat-square&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Qwen3.5--4B-111827?style=flat-square" alt="Qwen3.5-4B" />
  <img src="https://img.shields.io/badge/QLoRA-111827?style=flat-square" alt="QLoRA" />
  <img src="https://img.shields.io/badge/GAIA-111827?style=flat-square" alt="GAIA" />
  <img src="https://img.shields.io/badge/Local_First-111827?style=flat-square" alt="Local-first" />
</p>

<p align="center">
  <a href="#runtime">Runtime</a> ·
  <a href="#the-experiment">Experiment</a> ·
  <a href="#current-evidence">Evidence</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#evaluation-contract">Evaluation</a>
</p>

## At a glance

| | |
|---|---|
| **Question** | Can QLoRA improve the tool-use policy of a small local agent, or does it mostly add training cost? |
| **What I built** | A single-model Qwen3.5-4B agent with `search`, `read`, `inspect`, and `python`, plus controlled GAIA evaluation, trajectory collection, QLoRA training, leakage guards, and paired comparison. |
| **Runtime** | Local CLI with bounded tool I/O, explicit backend failure states, readiness checks, atomic run artifacts, and visible action traces. |
| **Current answer** | **Unanswered.** The experiment infrastructure exists, but no unseen GAIA result currently justifies a QLoRA improvement claim. |
| **Design focus** | Diagnose on exposed development data, promote only on blind evidence, then isolate runtime/policy gains from QLoRA gains. |

> [!NOTE]
> This repository treats **“QLoRA did not improve the frozen agent”** as a valid outcome. The goal is not to force a fine-tuning success story.

## Runtime

The runtime intentionally stays small: one model chooses a tool call or a final answer, and the harness owns execution boundaries around it.

```text
user task
   ↓
Qwen3.5-4B
   ↓
choose next action
   ├─ search
   ├─ read
   ├─ inspect
   └─ python
   ↓
observation
   ↓
next action or final answer
```

Operational behavior is explicit rather than hidden behind a larger agent framework:

- model timeouts, unavailable backends, capacity stops, and known model errors are recorded as distinct incomplete outcomes;
- tool exceptions become bounded observations so the model can change strategy;
- exact duplicate calls are blocked;
- URL reads block local/private destinations and stream within a source-size budget;
- workspace files are size-checked before expensive parsing;
- the Python tool runs in an isolated, time-limited subprocess with a restricted builtin surface;
- `small-agent doctor` checks local readiness without loading model weights;
- `small-agent run --output ...` writes an atomic `small-agent-run/v1` artifact with the answer, stop reason, metrics, trace, and effective runtime configuration.

This is a **production-style local runtime**, not a hardened multi-tenant service or hostile-code sandbox. There is no planner/router/critic graph.

## The experiment

The project asks two separate questions instead of mixing them together:

1. Can ordinary runtime/policy engineering improve the same 4B agent on unseen GAIA tasks?
2. After the best runtime/policy is frozen, does QLoRA add further value?

```text
Diagnostic25 (DEV, inspectable)
          ↓
one general hypothesis
          ↓
runtime/tool fix OR non-GAIA training treatment
          ↓
frozen candidate
          ↓
Blind Shadow25
     reject / promote
              ↓
      freeze final arms
              ↓
       Evaluation100
              ↓
Base vs policy vs policy+QLoRA
```

Each candidate changes one explainable treatment at a time. Correct GAIA answers are the primary metric; completion, tool success, latency, step count, and duplicate calls are diagnostics, not substitutes for accuracy.

## Current status

| Component | Status |
|---|---|
| Tool-using agent runtime | ✅ Implemented |
| Runtime readiness / failure contracts | ✅ Implemented |
| Diagnostic25 development partition | ✅ Implemented |
| Blind Shadow25 partition + promotion gate | ✅ Implemented |
| Aggregate-only matched Shadow workflow | ✅ Implemented |
| Frozen Evaluation100 partition | ✅ Implemented |
| Visible trajectory logging | ✅ Implemented |
| Verified non-GAIA trajectory collector | ✅ Implemented |
| Training-data leakage guard | ✅ Implemented |
| QLoRA training entry point | ✅ Implemented |
| Base-vs-LoRA comparator | ✅ Implemented |
| Shadow-promoted policy candidate | ⏳ Not yet established |
| Valid trained adapter for final comparison | ⏳ Not yet established |
| Final Base / policy / QLoRA result | ⏳ Not yet established |

## Current evidence

An earlier local diagnostic attempt stopped after **13 persisted tasks** because the machine crossed the RAM safety guard. It is preserved as development evidence, not presented as a benchmark result.

| Partial diagnostic fact | Result |
|---|---:|
| Persisted tasks | 13 / 25 |
| Coverage | 8 Level 1 / 5 Level 2 / 0 Level 3 |
| Exact match | 2 / 13 |
| Completed tasks | 6 / 13 |
| Model calls | 118 |
| Tool calls | 112 |
| Tool success | 81.25% |
| Stop reason | available RAM below guard threshold |

> [!IMPORTANT]
> The historical 13-case prefix is **DEV-only and distribution-biased** because truncation happened after level ordering. Repeated experiments on that prefix do not establish GAIA improvement. Later termination experiments that improved completion on exposed cases are likewise runtime/development observations unless they pass blind Shadow.

The preserved historical decision record is in [`docs/p4-decision.md`](docs/p4-decision.md).

## Evaluation contract

The pinned GAIA validation set is divided into three disjoint experiment roles:

```text
GAIA validation: 165

Diagnostic (DEV):    25   8 / 13 / 4 by level
Shadow (blind):      25   8 / 13 / 4 by level
Evaluation (FINAL): 100  32 / 52 / 16 by level
Unused:              15
```

### Diagnostic25

Diagnostic25 is the only GAIA set whose case-level questions, traces, answers, and failure signals may be inspected during development. It exists to discover general failure patterns and form falsifiable hypotheses. `--limit` is a debug convenience only and must not support a performance claim.

### Shadow25

Shadow25 is a blind promotion gate. Candidate and baseline run against the same deterministic manifest and matched model/runtime settings. Development sees aggregate metrics only; case-level questions, reference answers, model answers, traces, manifests, protected hashes, and task workspaces are not uploaded.

A candidate is promoted only when all three are true:

- exact correct count improves by at least **+2 tasks** over the frozen baseline;
- completion regresses by no more than **1 task**;
- no new catastrophic runtime stop class appears.

Efficiency improvements without an exact-correct improvement are runtime improvements, **not GAIA performance improvements**.

This cycle permits at most **three Shadow candidate exposures**. A rejected Shadow candidate is not debugged by opening Shadow cases.

### Evaluation100

Evaluation100 remains unopened during candidate development. Before it is opened, all final comparison arms must already be frozen. The intended final comparison separates:

1. frozen Base vs best Shadow-promoted runtime/policy;
2. the same best runtime/policy without vs with QLoRA.

This prevents using Evaluation100 feedback to decide how to tune the LoRA arm.

This is an **internal controlled evaluation setup**, not an official GAIA leaderboard submission.

## Training-data guard

Training remains non-GAIA. Diagnostic, Shadow, and Evaluation questions, answers, traces, and case-specific rules are forbidden from the QLoRA path.

GAIA questions are hashed locally and checked against candidate training inputs. Sources marked as GAIA are rejected. Exact hashes do not detect paraphrases, so provenance and manual judgment still matter.

The current independent curriculum is intentionally narrow (`read_fact`, `inspect_metadata`, `python_numeric`, `format_recovery`). It is extended only when development evidence identifies a general capability worth teaching. A low GAIA score by itself is not enough justification to train QLoRA.

Only trajectories that satisfy the verification contract are eligible for training: correct final answer, required tools satisfied, zero tool errors, zero duplicate blocks, complete provenance, and visible policy actions only.

## Quickstart

### Local runtime

For the ordinary Ollama-backed agent, install only the runtime tool extras:

```powershell
python -m pip install -e ".[search,files]"
```

With Ollama running and `qwen3.5:4b` available, check readiness before the first task:

```powershell
small-agent doctor
```

Then run a task and keep a reproducible JSON artifact:

```powershell
small-agent run "Find the relevant evidence and answer concisely." `
  --output runs/task/result.json
```

### Evaluation and training extras

```powershell
python -m pip install -e ".[eval,search,files]"
```

The Transformers / QLoRA path needs the heavier training dependencies and a compatible CUDA environment:

```powershell
python -m pip install -e ".[eval,search,files,train]"
small-agent doctor --backend transformers
```

### Development evaluation

Diagnostic25 is the inspectable development partition:

```powershell
small-agent gaia-eval `
  --partition diagnostic `
  --backend transformers `
  --work-root runs/gaia-diagnostic-base
```

`small-agent gaia-eval --partition shadow` exists for the controlled blind workflow; ordinary development should not use it interactively to inspect or tune Shadow cases.

The repository also retains the resource-aware local P4 pipeline for sealed local experiments. It is not a substitute for the blind Shadow promotion contract.

### Training

Collect verified non-GAIA trajectories:

```powershell
small-agent collect-trajectories `
  --tasks data/policy-tasks.jsonl `
  --output data/verified-trajectories.jsonl `
  --backend transformers `
  --protected-questions runs/gaia-protected-question-hashes.json
```

Train an adapter only after a training hypothesis is justified:

```powershell
small-agent train-qlora `
  --data data/verified-trajectories.jsonl `
  --output adapters/qwen3.5-4b-tool-policy `
  --protected-questions runs/gaia-protected-question-hashes.json
```

`small-agent compare-evals` remains the lower-level comparator for complete frozen Evaluation100 Base/+LoRA arms and rejects configuration drift other than the adapter treatment.

## What gets recorded

A single `small-agent run --output` artifact records:

- schema version and effective runtime configuration
- final answer, completion flag, and stop reason
- step/tool metrics
- visible action trace

The GAIA runner additionally records correctness, latency, capability-gap flags, diagnostic failure labels, and aggregate stop-reason counts.

Shadow CI publishes only the sanitized aggregate summary and promotion verdict. Raw gated task data stays ephemeral on the runner and is deleted before artifact upload.

## Repository shape

```text
small-agent-qlora/
├── src/gaia_small_agent/
│   ├── agent/
│   ├── model/
│   ├── tools/
│   ├── benchmark/
│   └── training/
├── docs/
└── tests/
```

## Limits

- The Python tool is process-isolated and time-limited, but not a hardened hostile-code sandbox.
- `inspect` does not provide semantic image / audio / video understanding.
- Search uses the live web, so exact results can change; matched baseline/candidate runs reduce but do not eliminate this variance.
- Runtime source-size limits reduce accidental memory blow-ups; they are not a complete security boundary.
- Exact hash checks do not catch paraphrased benchmark leakage.
- The current independent synthetic curriculum is deliberately small and does not cover every autonomous tool-selection pattern.
- The Transformers / QLoRA path needs a compatible local CUDA / PyTorch / bitsandbytes setup.
- No final Base-vs-LoRA claim should be made until the final arms are frozen and a valid paired Evaluation100 run completes.

## Verification

CPU-only CI verifies package installation, CLI entry-point loading, tests, and compilation on Linux and Windows. Real model/GAIA evidence is a separate controlled experiment contract; passing CI is not a benchmark result.

The full framework design is in [`docs/superpowers/specs/2026-10-09-gaia-improvement-framework-design.md`](docs/superpowers/specs/2026-10-09-gaia-improvement-framework-design.md).

## References

- [Pi minimal harness](https://pi.dev/)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- [GAIA scorer](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
