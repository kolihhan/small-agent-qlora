# GAIA Improvement Framework

## Goal

Turn Small-Agent into a controlled R&D system for improving a local Qwen3.5-4B tool agent on GAIA while preserving a defensible answer to the repository's core question: **what actually improves unseen GAIA performance, and does QLoRA add value beyond runtime/policy engineering?**

The framework must support repeated improvement cycles without turning GAIA into a memorized development set. It reuses the repository's existing runtime, evaluator, trajectory collector, QLoRA trainer, and comparator instead of introducing a new agent framework.

## Current System to Reuse

The repository already contains the needed foundations:

- one Qwen3.5-4B agent runtime with `search`, `read`, `inspect`, and `python`;
- bounded tool observations, duplicate-call blocking, visible traces, and explicit runtime stop reasons;
- deterministic GAIA diagnostic/evaluation selection at a pinned dataset revision;
- exact GAIA scoring plus completion/tool/failure metrics;
- verified non-GAIA trajectory collection with provenance and leakage guards;
- QLoRA training from verified trajectories only;
- Base-vs-LoRA paired comparison on complete Evaluation100 runs.

The framework organizes and gates those capabilities. It does not replace them.

## Design Principles

1. **One hypothesis per candidate.** A candidate represents one explainable treatment: one runtime/policy change, one tool-quality change, or one training treatment.
2. **Correctness first.** Exact GAIA correct count is the primary performance metric. Completion, tool success, latency, steps, and duplicates are secondary diagnostics.
3. **Generalize before promote.** Development evidence may generate a hypothesis, but unseen evidence decides whether it survives.
4. **Training stays non-GAIA.** GAIA questions, answers, traces, entities, and case-specific rules never become training data.
5. **Smallest intervention first.** Prefer evaluator, tool/runtime, or small generic policy fixes before training or architecture.
6. **Blind-data exposure is budgeted.** Aggregate Shadow scores can also cause adaptive overfitting if queried repeatedly.
7. **Final means final.** All final experimental arms are frozen before Evaluation100 is opened.
8. **Negative results count.** A result that QLoRA or a policy idea does not improve unseen GAIA is valid.

## Architecture

```text
Base Agent
    |
    v
Diagnostic25 (DEV)
    |
    v
Failure Analysis
    |
    v
One General Hypothesis
    |
    +----------------------+
    |                      |
    v                      v
Runtime / Tool Fix     Training Hypothesis
    |                      |
    |                 Non-GAIA Curriculum
    |                      |
    |                 Verified Trajectories
    |                      |
    |                     QLoRA
    |                      |
    +----------+-----------+
               |
               v
        Frozen Candidate
               |
               v
      Blind Shadow25 Gate
          |           |
        reject      promote
                       |
                       v
             Freeze Final Arms
          Base / Policy / LoRA
                       |
                       v
                Evaluation100
                       |
                       v
          Engineering gain + LoRA gain
```

The system has five conceptual modules only: **Runtime, Eval, Diagnose, Improve, Compare**. These are workflow roles, not new services.

## 1. Runtime

The runtime remains a single model choosing either one of the existing tools or a final answer. The tool surface stays fixed during a controlled comparison unless the experiment explicitly tests a tool change.

Runtime changes are eligible only when they solve a generic failure pattern. Examples include termination behavior, duplicate-loop prevention, malformed-final recovery, search/read behavior, or tool boundary bugs. A change that encodes knowledge from one GAIA case is forbidden.

Runtime quality and GAIA accuracy are reported separately. A candidate may be retained as a runtime improvement even when it does not improve GAIA correctness, but it must not be described as a GAIA performance win.

## 2. Evaluation

### Diagnostic25 — development set

Diagnostic25 is the only GAIA partition whose case-level content may be inspected during development.

- Run all 25 cases for candidate-level development evidence.
- Questions, traces, answers, failure signals, and tool behavior may be inspected.
- `--limit` is debug-only and must never support a performance claim.
- Historical first-13 runs are development artifacts only. They are additionally distribution-biased because diagnostic tasks are level-ordered before truncation.

Diagnostic25 discovers broad failure patterns. It does not establish generalization.

### Shadow25 — blind validation gate

Create one deterministic Shadow25 from GAIA validation rows belonging to neither Diagnostic25 nor Evaluation100.

Preferred quotas match Diagnostic25: Level 1/2/3 = `8/13/4`. The implementation fails loudly if the pinned dataset revision cannot supply those quotas after existing partitions are reserved.

Shadow rules:

- deterministic manifest at the pinned GAIA revision;
- pairwise disjoint from Diagnostic25 and Evaluation100;
- candidate frozen before running Shadow;
- no case-level question, reference answer, model answer, or trace exposed to development;
- CI emits aggregate metrics and protocol metadata only;
- baseline and candidate run in the same workflow against the same model digest and manifest when practical.

Promotion gate:

- exact correct count improves by at least **+2 tasks** versus frozen baseline;
- completion regresses by at most **1 task**;
- no new catastrophic runtime failure class;
- efficiency metrics cannot substitute for correctness.

#### Shadow exposure budget

A blind set can still be overfit through repeated aggregate queries. Therefore:

- at most **3 candidate generations total** may query Shadow25 before the final Evaluation100 cycle;
- every Shadow query is logged with candidate SHA and predeclared hypothesis;
- a failed candidate cannot be modified and immediately retried as the same hypothesis; a retry consumes another generation and must be justified by new DEV/non-GAIA evidence;
- Shadow case details remain sealed even after a failure;
- when the budget is exhausted, policy development stops. No new Shadow partition is invented to continue tuning.

The budget is intentionally small. It is a project-level guardrail, not a statistical guarantee.

### Evaluation100 — final set

Evaluation100 is opened only after **all final arms are frozen**.

The final sealed arms are:

- **A — Original Base:** frozen reference runtime + base model;
- **B — Best Policy:** promoted runtime/policy + the same base model;
- **C — Best Policy + QLoRA:** exactly B's runtime/policy plus the frozen adapter.

No curriculum, prompt, runtime, tool, hyperparameter, or adapter decision may be changed after any Evaluation100 result is observed.

All three final arms use the same pinned GAIA manifest and, for the scientific comparison, the same Transformers model snapshot/backend family so the LoRA arm is directly comparable. Evaluation100 task-level transitions are retained for final interpretation but are not fed back into development.

This gives two clean final effects:

1. **B - A:** runtime/policy engineering contribution;
2. **C - B:** QLoRA contribution.

## 3. Diagnosis

Diagnosis happens only on Diagnostic25 and ordinary non-GAIA regression tasks.

Use one stable taxonomy:

```text
Tool Selection
  - wrong tool / unnecessary tool / premature final
Tool Execution
  - bad query / bad arguments / tool error recovery
Evidence Acquisition
  - missing evidence / wrong evidence / insufficient evidence accepted
Reasoning & Synthesis
  - evidence correct but conclusion wrong / computation wrong / multi-step loss
Control
  - duplicate loop / excessive steps / termination failure
Final Answer
  - empty / malformed / verbosity / exact-answer formatting
Capability Gap
  - unsupported semantic input or genuinely missing tool capability
Infrastructure
  - model/backend/resource/network failure that invalidates the experiment
```

Failure labels are signals, not automatic causal truth. A hypothesis must be supported by multiple DEV cases or by a reproducible non-GAIA regression before implementation.

Each candidate records only what is needed:

- observed failure cluster;
- one falsifiable hypothesis;
- single treatment;
- predicted metric change;
- Diagnostic verdict;
- Shadow verdict if queried.

A small Markdown/JSON experiment record is enough. No experiment database is needed.

## 4. Improvement Ladder

Every improvement cycle starts at the cheapest plausible layer and stops at the first layer that addresses the general problem:

1. evaluator/scoring/protocol bug;
2. tool or runtime correctness bug;
3. small generic policy/prompt/control change;
4. search/read/inspect/python quality change;
5. non-GAIA curriculum change;
6. QLoRA;
7. agent architecture change only if earlier layers fail and evidence requires it.

Do not combine treatments in one candidate. If two changes are desirable, promote the first independently before testing the second.

## 5. Non-GAIA Improvement and Training

The current curriculum is deliberately narrow (`read_fact`, `inspect_metadata`, `python_numeric`, `format_recovery`). It is a starting regression/training source, not a proxy for GAIA.

When Diagnostic25 exposes a general policy weakness, translate it into a **domain-independent capability**, then create independent non-GAIA tasks for that capability.

Example:

```text
GAIA observation:
  evidence was gathered correctly but final synthesis was wrong

Allowed capability:
  multiple unrelated facts -> combine -> concise exact answer

Forbidden:
  reproduce/paraphrase the GAIA question, answer, entities, or case-specific route
```

Training-data contract stays strict:

- provenance complete;
- non-GAIA source;
- no protected-question hash overlap;
- correct final answer;
- required tools satisfied;
- zero tool errors;
- zero duplicate blocks;
- visible policy trajectory only; no hidden chain-of-thought.

QLoRA is justified only after there is a clear general policy capability to teach and enough verified trajectories to support it. "GAIA score is low" is not a training hypothesis.

The initial framework does **not** add a stronger teacher, hand-authored policy generator, or new data engine. If the existing collector cannot produce enough verified examples for an evidence-backed capability, that limitation is recorded and a separate design decision is required.

## 6. Candidate Lifecycle

```text
HYPOTHESIS
   -> DEV VERIFIED
   -> FROZEN
   -> SHADOW PASS / SHADOW REJECT
   -> FINAL-ARM FROZEN
   -> FINAL EVALUATED
```

- `HYPOTHESIS`: treatment and falsifiable prediction are written before implementation/result inspection.
- `DEV VERIFIED`: unit/regression checks pass and Diagnostic25 evidence is consistent with the hypothesis.
- `FROZEN`: candidate SHA and complete runtime config are fixed.
- `SHADOW PASS`: predeclared aggregate gate passes.
- `SHADOW REJECT`: direction stops unless a genuinely new DEV-backed generation spends another Shadow exposure.
- `FINAL-ARM FROZEN`: A/B/C identities, code, model snapshot, adapter, and configs are sealed before Evaluation100.
- `FINAL EVALUATED`: complete Evaluation100 results exist for all required arms.

Changing behavior after freezing creates a new candidate generation.

## 7. Backend and Comparison Discipline

GitHub-hosted Ollama is useful for inexpensive, reproducible DEV/Shadow screening. QLoRA evaluation requires the Transformers backend because the current adapter path is implemented there.

Therefore:

- DEV/Shadow policy screening may use Ollama when baseline and candidate are matched within that screen;
- before final freeze, the promoted policy must pass ordinary tests and a final-backend readiness/parity check on DEV/non-GAIA data;
- Evaluation100 A/B/C runs use the same pinned Transformers base model snapshot and compatible runtime configuration;
- B vs C changes only the adapter treatment;
- A vs B changes only the promoted policy/runtime treatment being claimed.

Backend differences are never counted as model improvement.

## 8. Final Comparison Matrix

The final report should reduce to one compact table:

| Version | Diagnostic25 | Shadow25 | Evaluation100 | Avg calls/task | Interpretation |
|---|---:|---:|---:|---:|---|
| A Original Base | DEV reference | blind reference | final | x | baseline |
| B Best policy/runtime | DEV | must pass | final | x | engineering contribution |
| C B + QLoRA | non-GAIA training checks | frozen before final | final | x | training contribution |

The project answers:

- Did engineering improve unseen GAIA? `B - A`
- Did QLoRA add value after engineering? `C - B`

A negative or zero delta is reported as-is.

## 9. Workflow and Artifact Rules

Use GitHub Actions for reproducible Ollama-backed evaluation where practical.

For gated GAIA content:

- never publish questions/reference answers;
- Diagnostic case-level artifacts stay private to the development experiment;
- Shadow artifacts are aggregate-only;
- final public docs report only permitted metrics/protocol metadata;
- record model identity/digest, candidate SHA, dataset revision, seed, step/token limits, thinking mode, tool surface, and Shadow exposure count.

Live search introduces environmental variance. Matched baseline/candidate runs are preferred. Do not build web record/replay infrastructure unless observed variance materially prevents interpretation.

## 10. Minimal Code Surface

Ponytail rule: reuse existing modules and change as few files as possible.

Initial framework implementation should require only:

- `benchmark/gaia100.py`: deterministic `shadow` selection while preserving existing Diagnostic25/Evaluation100;
- `cli.py`: allow `--partition shadow`;
- focused partition/manifest tests;
- one aggregate-only paired Shadow workflow with the predeclared gate and exposure metadata;
- one lightweight experiment-record convention using current JSON/Markdown artifacts;
- README/protocol documentation that makes DEV -> Shadow -> Final and A/B/C explicit.

Do **not** initially add:

- a new orchestration package;
- planner/critic/router/memory services;
- vector DB;
- MLflow or experiment database;
- generic plugin system;
- new training architecture;
- web replay infrastructure.

Curriculum expansion and QLoRA changes are separate evidence-gated candidates, not automatic framework setup work.

## 11. Immediate Reset from Current State

Existing first-13 experiments remain DEV observations only. They are not benchmark improvements.

Current termination/empty-final branches may be retained as candidate evidence, but they must re-enter this lifecycle and earn promotion on blind Shadow before any GAIA improvement claim.

No further patch should be driven by individual first-13 outcomes before the evaluation framework is implemented.

## Explicit Non-Goals

- no official GAIA leaderboard claim;
- no memorization or paraphrased training from GAIA;
- no case-level tuning on Shadow or Evaluation100;
- no unlimited aggregate probing of Shadow;
- no automatic architecture search;
- no multi-agent system by default;
- no requirement that QLoRA must win;
- no complexity solely for production appearance.

## Success Criteria

The framework is complete when:

1. Diagnostic25, Shadow25, and Evaluation100 are deterministic and pairwise disjoint at the pinned GAIA revision.
2. Diagnostic25 can generate one falsifiable general hypothesis without touching blind data.
3. A candidate can be frozen with full model/runtime identity and evaluated on Shadow using aggregate-only output.
4. Shadow exposure is counted and capped at three candidate generations.
5. A Shadow failure does not expose case details or silently create a replacement holdout.
6. Final A/B/C arms are all frozen before Evaluation100 is opened.
7. Evaluation100 uses a comparable Transformers base snapshot so `B-A` and `C-B` have clear meanings.
8. Non-GAIA curricula can feed the existing verified trajectory collector and QLoRA trainer without GAIA leakage.
9. Final evidence separately answers whether policy/runtime engineering helped and whether QLoRA added further value.
10. The framework remains valid if no candidate improves or if QLoRA does not help.
