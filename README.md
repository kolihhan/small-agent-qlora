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
| **Question** | What actually improves unseen GAIA performance for this local 4B agent, and does QLoRA add value beyond runtime/policy engineering? |
| **What I built** | A single-model Qwen3.5-4B agent with `search`, `read`, `inspect`, and `python`, plus controlled GAIA evaluation, trajectory collection, QLoRA training, leakage guards, and paired comparison. |
| **Runtime** | Local CLI with bounded tool I/O, explicit backend failure states, readiness checks, atomic run artifacts, and visible action traces. |
| **Current answer** | **Unanswered.** The improvement framework exists, but there is no valid unseen GAIA improvement claim or Base-vs-LoRA result yet. |
| **Design focus** | One hypothesis per candidate, blind promotion before final evaluation, and honest stop conditions under local hardware constraints. |

> [!NOTE]
> This repository treats **“the policy change did not generalize”** and **“QLoRA was not justified by the available evidence”** as valid outcomes. The goal is not to force an improvement story.

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

The project separates engineering gains from fine-tuning gains instead of assuming QLoRA is the answer.

```text
Base agent
   ↓
Diagnostic25 (inspectable DEV)
   ↓
one general hypothesis
   ↓
frozen candidate
   ↓
Blind Shadow25
   ├─ reject → stop that direction
   └─ pass   → freeze final arms
                  ↓
             Evaluation100
                  ↓
        Base vs policy vs policy+QLoRA
```

Controlled comparisons keep the base model family, scorer, tool surface, step/token limits, and evaluation manifest matched unless that exact variable is the treatment under test.

## Current status

| Component | Status |
|---|---|
| Tool-using agent runtime | ✅ Implemented |
| Runtime readiness / failure contracts | ✅ Implemented |
| Diagnostic25 / Evaluation100 runner | ✅ Implemented |
| Deterministic blind Shadow25 selector | ✅ Implemented |
| Aggregate-only Shadow promotion gate | ✅ Implemented |
| Visible trajectory logging | ✅ Implemented |
| Verified trajectory collector | ✅ Implemented |
| Training-data leakage guard | ✅ Implemented |
| QLoRA training entry point | ✅ Implemented |
| Base-vs-LoRA comparator | ✅ Implemented |
| Shadow-passing policy candidate | ⏳ Not yet established |
| Valid trained adapter for final comparison | ⏳ Not yet established |
| Final unseen GAIA result | ⏳ Not yet established |

## Current evidence

An earlier diagnostic attempt stopped after **13 persisted tasks** because the machine crossed the RAM safety guard. It is preserved as debugging evidence, not presented as a complete benchmark.

| Partial diagnostic fact | Result |
|---|---:|
| Persisted tasks | 13 / 25 |
| Exact match | 2 / 13 |
| Completed tasks | 6 / 13 |
| Model calls | 118 |
| Tool calls | 112 |
| Tool success | 81.25% |
| Stop reason | available RAM below guard threshold |

> [!IMPORTANT]
> These 13 tasks are an exposed **DEV prefix**, not a GAIA performance result. The diagnostic selection is level-ordered before truncation, so `--limit 13` is also distribution-biased and contains no Level 3 coverage. Historical first-13 experiments may inform debugging only; they cannot establish generalization or justify QLoRA.

The preserved historical decision record is in [`docs/p4-decision.md`](docs/p4-decision.md).

## Evaluation contract

The pinned GAIA validation set is reserved into three roles:

```text
Diagnostic25   inspectable DEV
Shadow25       blind promotion gate
Evaluation100  final measurement
```

The exact Shadow quotas are `8 / 13 / 4` across GAIA Levels 1 / 2 / 3. Selection is deterministic and must be disjoint from the existing Diagnostic25 and Evaluation100 partitions; the selector fails rather than silently changing quotas if the pinned dataset cannot satisfy the reservation.

The contract is:

1. Inspect **all Diagnostic25** tasks to identify a broad failure pattern. `--limit` is debug-only and never supports a performance claim.
2. State one falsifiable hypothesis and change one treatment at a time.
3. Freeze the candidate before blind evaluation.
4. Compare baseline and candidate on the same **Shadow25** manifest. Shadow output is aggregate-only; case-level Shadow questions, answers, traces, and task IDs are not used for tuning.
5. Promote only if the candidate gains at least **+2 exact-correct tasks**, completion regresses by no more than one task, and no new catastrophic runtime failure class appears. Better latency/tool efficiency alone is not a GAIA performance win.
6. Limit this cycle to at most **three Shadow exposures**. A failed Shadow candidate is not debugged against Shadow cases.
7. Before opening Evaluation100, freeze the final arms: Base, best policy/runtime candidate, and the same policy/runtime plus QLoRA if training is justified.
8. Run the frozen **Evaluation100** arms and compare task-level transitions. Do not tune after seeing final results.

This is an **internal controlled evaluation setup**, not an official GAIA leaderboard submission.

## Improvement and training

The improvement ladder is deliberately boring:

1. evaluator/protocol bug;
2. tool or runtime correctness bug;
3. small generic policy/control change;
4. tool-quality change;
5. non-GAIA curriculum change;
6. QLoRA;
7. architecture change only if earlier layers fail and evidence requires it.

The current synthetic curriculum is intentionally narrow. When Diagnostic25 exposes a general policy weakness, it may motivate new **independent non-GAIA** tasks that teach that abstract capability. GAIA questions, answers, traces, entities, and case-specific routes are forbidden from training data.

Only verified trajectories are training-eligible: correct final answer, required tools satisfied, zero tool errors, zero duplicate blocks, complete provenance, and no protected GAIA question overlap.

QLoRA is therefore evidence-gated. A low GAIA score by itself is not a reason to train.

## Training-data guard

GAIA questions are hashed locally and checked against candidate training inputs. Sources marked as GAIA are rejected from the training path.

This catches exact reuse. It **does not** detect a paraphrased benchmark question, so provenance still requires manual judgment.

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

The run artifact keeps the existing result fields and also records the backend/model identity, adapter path when applicable, step/token limits, thinking mode, tool surface, and observation cap.

### Evaluation and training extras

Install evaluation support without the training stack when that is all you need:

```powershell
python -m pip install -e ".[eval,search,files]"
```

The Transformers / QLoRA path needs the heavier training dependencies and a compatible CUDA environment:

```powershell
python -m pip install -e ".[eval,search,files,train]"
small-agent doctor --backend transformers
```

### Controlled improvement workflow

Development uses Diagnostic25. Blind promotion is performed by `.github/workflows/gaia-shadow-gate.yml`, which runs baseline and candidate on the same GitHub runner and uploads only one sanitized aggregate verdict. Evaluation100 remains unopened until the final experiment arms are frozen.

The older `scripts/run_p4_full_pipeline.ps1` is preserved for historical reproducibility; it is not the current promotion protocol.

### Development commands

Run the inspectable diagnostic partition:

```powershell
small-agent gaia-eval `
  --partition diagnostic `
  --backend transformers `
  --work-root runs/gaia-diagnostic-base
```

`--partition shadow` exists for the controlled blind gate. Do not use it interactively to inspect or tune individual Shadow cases.

Collect verified training trajectories:

```powershell
small-agent collect-trajectories `
  --tasks data/policy-tasks.jsonl `
  --output data/verified-trajectories.jsonl `
  --backend transformers `
  --protected-questions runs/gaia-protected-question-hashes.json
```

Train an adapter only when a training hypothesis is justified:

```powershell
small-agent train-qlora `
  --data data/verified-trajectories.jsonl `
  --output adapters/qwen3.5-4b-tool-policy `
  --protected-questions runs/gaia-protected-question-hashes.json
```

`small-agent compare-evals` is a lower-level comparator for complete frozen Evaluation100 Base-vs-LoRA arms. It rejects partial runs and configuration drift other than the adapter treatment.

## What gets recorded

A single `small-agent run --output` artifact records:

- schema version and effective runtime configuration
- final answer, completion flag, and stop reason
- step/tool metrics
- visible action trace

The GAIA runner additionally records correctness, latency, capability-gap flags, failure labels, failure signals, and aggregate stop reasons.

Diagnostic case-level artifacts are development evidence. The blind Shadow workflow does **not** upload raw results, manifests, task IDs, traces, protected hashes, or workspaces; only its aggregate gate summary is retained.

It flags cases with known unsupported semantic image / audio / video attachments. That flag describes the current tool surface; it is not proof that the attachment was semantically necessary to answer the task.

The current `inspect` tool reads document and tabular structure; it is **not** a general vision or audio model.

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
- Search uses the live web, so exact results can change; matched baseline/candidate Shadow runs reduce but do not eliminate this variance.
- Runtime source-size limits reduce accidental memory blow-ups; they are not a complete security boundary.
- Exact hash checks do not catch paraphrased benchmark leakage.
- The current independent synthetic curriculum is deliberately small and does not cover every autonomous tool-selection pattern.
- The Transformers / QLoRA path needs a compatible local CUDA / PyTorch / bitsandbytes setup.
- No final GAIA or Base-vs-LoRA claim should be made until the frozen final experiment completes.

## Verification

CPU-only CI verifies package installation, CLI entry-point loading, tests, and compilation on Linux and Windows. The blind Shadow workflow is a separate evidence gate; passing ordinary CI is not a benchmark result.

See [`docs/verification.md`](docs/verification.md) for the distinction between repository verification and preserved experiment evidence.

## References

- [Pi minimal harness](https://pi.dev/)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- [GAIA scorer](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
