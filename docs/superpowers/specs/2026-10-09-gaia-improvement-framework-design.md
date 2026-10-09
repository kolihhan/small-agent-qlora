# GAIA Improvement Framework

## Goal

Turn Small-Agent into a controlled R&D system for improving a local Qwen3.5-4B tool agent on GAIA while preserving a defensible answer to the repository's core question: **what actually improves unseen GAIA performance, and does QLoRA add value beyond runtime/policy engineering?**

The framework must support repeated improvement cycles without turning GAIA into a memorized development set, and it must reuse the repository's existing runtime, evaluator, trajectory collector, trainer, and comparator instead of introducing a new agent framework.

## Current System to Reuse

The repository already contains the needed foundations:

- one Qwen3.5-4B agent runtime with `search`, `read`, `inspect`, and `python`;
- bounded tool observations, duplicate-call blocking, visible traces, and explicit runtime stop reasons;
- deterministic GAIA diagnostic/evaluation selection at a pinned dataset revision;
- exact GAIA scoring plus completion/tool/failure metrics;
- verified non-GAIA trajectory collection with provenance and leakage guards;
- QLoRA training from verified trajectories only;
- Base-vs-LoRA paired comparison on complete Evaluation100 runs.

The framework therefore organizes and gates existing capabilities. It does not replace them.

## Design Principles

1. **One hypothesis per candidate.** A candidate must represent one explainable treatment: one runtime/policy change, one tool-quality change, or one training treatment.
2. **Correctness first.** Exact GAIA correct count is the primary performance metric. Completion, tool success, latency, steps, and duplicates are secondary diagnostics.
3. **Generalize before promote.** Development evidence may generate a hypothesis, but unseen evidence decides whether the hypothesis survives.
4. **Training stays non-GAIA.** GAIA questions, answers, traces, and case-specific rules never become training data.
5. **Smallest intervention first.** Prefer evaluation bug fixes, tool/runtime fixes, or a small generic policy change before new training or architecture.
6. **No score chasing.** Once a blind partition is observed, it is not opened for case-level tuning.
7. **Negative results count.** A result that QLoRA or a policy idea does not improve unseen GAIA is a valid project result.

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
                Evaluation100
                       |
                       v
             Base / Policy / LoRA
                 paired result
```

The system has five conceptual modules only: **Runtime, Eval, Diagnose, Improve, Compare**. These are workflow roles, not five new services or frameworks.

## 1. Runtime

The runtime remains a single model choosing either one of the existing tools or a final answer. The current tool surface stays fixed during a controlled comparison unless the experiment explicitly tests a tool change.

Runtime changes are eligible only when they solve a generic failure pattern. Examples include termination behavior, duplicate-loop prevention, malformed final recovery, or tool boundary bugs. A change that encodes knowledge from one GAIA case is forbidden.

Runtime hardening and GAIA accuracy are reported separately. A candidate can be retained as a runtime-quality improvement even when it does not improve GAIA correctness, but it must not be presented as a GAIA performance win.

## 2. Evaluation

### Diagnostic25 — development set

Diagnostic25 is the only GAIA partition whose case-level content may be inspected during development.

- Run all 25 cases for candidate-level development evidence.
- Questions, traces, answers, failure signals, and tool behavior may be inspected.
- `--limit` is debug-only and must never support a performance claim.
- Historical first-13 runs are development artifacts only. They are additionally distribution-biased because the selected diagnostic tasks are level-ordered before truncation.

Diagnostic25 is used to discover broad failure patterns, not to decide that a candidate generalizes.

### Shadow25 — blind validation gate

Create one deterministic Shadow25 from GAIA validation rows that belong to neither Diagnostic25 nor Evaluation100.

Preferred quotas are the same as Diagnostic25: Level 1/2/3 = `8/13/4`. The implementation must fail loudly if the pinned dataset revision cannot supply those quotas after the existing partitions are reserved.

Shadow rules:

- deterministic manifest at the pinned GAIA revision;
- pairwise disjoint from Diagnostic25 and Evaluation100;
- candidate is frozen before running Shadow;
- case-level question, reference answer, model answer, and trace are not exposed to the development loop;
- CI may emit only aggregate metrics and protocol metadata;
- baseline and candidate run in the same workflow against the same model digest and manifest when possible.

Promotion gate:

- exact correct count must improve by at least **+2 tasks** versus the frozen baseline;
- completion may regress by at most **1 task**;
- no new catastrophic runtime failure class;
- efficiency metrics do not substitute for correctness.

A Shadow failure rejects that candidate direction. Shadow cases are not opened to explain the failure.

### Evaluation100 — final set

Evaluation100 remains the final frozen measurement.

- It is not inspected during runtime/policy development.
- Only a Shadow-passing candidate reaches it.
- Baseline and candidate use matched configuration except for the treatment under test.
- Task-level transitions (`wrong -> correct`, `correct -> wrong`) are retained for final scientific interpretation.

The pre-QLoRA best policy candidate establishes the final Base-policy result. QLoRA is evaluated later against the same frozen Evaluation100 manifest.

## 3. Diagnosis

Diagnosis happens only on Diagnostic25 and ordinary non-GAIA regression tasks.

Use one stable taxonomy to turn case failures into general hypotheses:

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

Failure labels are signals, not automatic causal truth. A hypothesis must be supported by multiple cases or by a reproducible non-GAIA regression before implementation.

Each candidate records:

- the observed failure cluster;
- one falsifiable hypothesis;
- the single treatment being applied;
- expected change in metrics;
- actual Diagnostic and Shadow verdict.

A Markdown/JSON experiment record is sufficient. No MLflow or experiment database is needed.

## 4. Improvement Ladder

Every improvement cycle starts at the cheapest plausible layer and stops at the first layer that solves the general problem:

1. evaluator/scoring/protocol bug;
2. tool or runtime correctness bug;
3. small generic policy/prompt/control change;
4. search/read/inspect/python quality change;
5. non-GAIA curriculum change;
6. QLoRA;
7. agent architecture change only if earlier layers fail and evidence requires it.

Do not combine multiple treatments in one candidate. If two changes are both desirable, promote the first independently before testing the second.

## 5. Non-GAIA Improvement and Training

The current curriculum is deliberately narrow (`read_fact`, `inspect_metadata`, `python_numeric`, `format_recovery`). It remains a starting regression/training source, not a proxy for GAIA.

When Diagnostic25 reveals a general policy weakness, translate the weakness into a **domain-independent capability**, then create independent non-GAIA tasks for that capability.

Example:

```text
GAIA observation:
  evidence was gathered correctly but the final synthesis was wrong

Allowed training abstraction:
  multiple unrelated local facts -> combine -> concise exact answer

Forbidden training abstraction:
  reproduce or paraphrase the GAIA question, answer, entities, or case-specific route
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

QLoRA is justified only after there is a clear general policy capability to teach and enough verified trajectories to support it. Training because "the GAIA score is low" is not sufficient justification.

## 6. Candidate Lifecycle

A candidate moves through this state machine:

```text
HYPOTHESIS
   -> DEV VERIFIED
   -> FROZEN
   -> SHADOW PASS / SHADOW REJECT
   -> FINAL EVALUATED
```

Rules:

- `HYPOTHESIS`: treatment and success prediction are written before code/result inspection.
- `DEV VERIFIED`: unit/regression checks pass and Diagnostic25 evidence is consistent with the hypothesis.
- `FROZEN`: candidate SHA and complete runtime configuration are fixed.
- `SHADOW PASS`: satisfies the predeclared aggregate gate.
- `SHADOW REJECT`: direction stops; do not inspect Shadow cases and patch toward them.
- `FINAL EVALUATED`: complete Evaluation100 paired result exists.

Changing behavior after freezing creates a new candidate generation.

## 7. Comparison Matrix

The final project should support a compact comparison such as:

| Version | Diagnostic25 | Shadow25 | Evaluation100 | Avg calls/task | Interpretation |
|---|---:|---:|---:|---:|---|
| Base | dev | frozen baseline | final baseline | x | reference |
| Best policy/runtime | dev | must pass | final | x | engineering contribution |
| Best policy/runtime + QLoRA | training regression only | optional pre-final safety gate | final | x | training contribution |

The important scientific comparisons are:

1. **Base vs best policy/runtime** — did engineering improve unseen GAIA?
2. **Best policy/runtime vs same runtime + QLoRA** — did QLoRA add value?

Do not compare unrelated agents or change multiple runtime settings between those arms.

## 8. Workflow and Artifact Rules

Use GitHub Actions for reproducible Ollama-backed evaluation where practical, because the repository has already demonstrated real Qwen inference on GitHub-hosted runners.

For gated GAIA content:

- never publish questions/reference answers;
- Diagnostic case-level artifacts stay local/private to the experiment context;
- Shadow artifacts are aggregate-only;
- final public documentation reports only metrics/protocol metadata permitted by the dataset contract;
- model identity/digest, candidate SHA, dataset revision, seed, step/token limits, thinking mode, and tool surface are recorded.

Live search introduces environmental variance. Matched baseline/candidate runs are preferred. Do not build web record/replay infrastructure unless observed variance materially prevents interpretation.

## 9. Minimal Code Surface

Ponytail rule: reuse the existing modules and change as few files as possible.

Expected additions/changes:

- extend `benchmark/gaia100.py` to support deterministic `shadow` selection while preserving Diagnostic25 and Evaluation100;
- allow `--partition shadow` in `cli.py`;
- add tests for deterministic/disjoint quotas and manifest mismatch;
- add one aggregate-only paired Shadow workflow;
- add one lightweight experiment record format or document convention using existing JSON/Markdown artifacts;
- update README/protocol docs so first-13 results are explicitly DEV-only and the full improvement loop is visible.

Do **not** create a new orchestration package, planner, critic, router, memory store, vector database, MLflow integration, experiment database, or generic plugin system.

Curriculum expansion and QLoRA changes happen only when an accepted hypothesis reaches the training layer; they are not part of the initial protocol implementation by default.

## 10. Immediate Reset from Current State

Existing first-13 policy experiments remain useful only as development observations. They must not be promoted as benchmark improvement.

The current termination/empty-final candidates may be retained as candidate evidence, but they must enter the new framework from `HYPOTHESIS/DEV VERIFIED` and earn promotion on blind Shadow before any claim of GAIA improvement.

No further patching should be driven by individual first-13 outcomes before the evaluation framework is in place.

## Explicit Non-Goals

- no official GAIA leaderboard claim;
- no memorization or paraphrased training from GAIA;
- no tuning on Shadow or Evaluation100;
- no automatic architecture search;
- no multi-agent system by default;
- no requirement that QLoRA must win;
- no complexity added solely to make the project look production-grade.

## Success Criteria

The framework is successful when all of the following are true:

1. Diagnostic25, Shadow25, and Evaluation100 are deterministic and pairwise disjoint at the pinned GAIA revision.
2. A developer can inspect Diagnostic25, state one general hypothesis, produce one candidate, and freeze it without touching blind data.
3. CI can compare frozen baseline vs candidate on Shadow with aggregate-only output and an automatic promotion verdict.
4. A Shadow failure stops that candidate direction without exposing case-level Shadow data.
5. A Shadow pass can produce a complete paired Evaluation100 result.
6. Non-GAIA capability curricula can feed the existing verified-trajectory collector and QLoRA trainer without GAIA leakage.
7. Final evidence can separately answer whether runtime/policy engineering helped and whether QLoRA added further value.
8. The result remains valid even if the answer is "no candidate improved" or "QLoRA did not help."
