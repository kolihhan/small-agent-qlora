# Qwen3.5-9B Adapter versus Scaffold: frozen 2×2 research gate

Status: **PREREGISTRATION ONLY / NO PHYSICAL EXPERIMENT / NO GAIA PROMOTION** (2026-11-17).

## Competing hypotheses

- **Hweights:** training v3B changes genuine multi-hop task solving independent of output postprocessing.
- **Hserialization:** publicly observed QA exactness differences mainly arise from a gold-blind same-base-model verbatim final extractor.
- **Hinteraction:** the serializer benefits only the adapter (or only the base), so naive independent arm comparisons misattribute the gain.
- **Hnull:** timeouts, source selection, stochastic decoding or small sample explain both effects.

## Controls and 2×2

Before inference, freeze source SHA, 24 unique never-before-tested non-GAIA task identities across ≥2 distinct benchmark/task families, answer-support-path overlap audit and total exposure ledger. If source isolation cannot be verified: **BLOCKED**.

| | Native final | Frozen verbatim serializer |
|---|---|---|
| Base (same Qwen3.5-9B digest) | base_native | base_serializer |
| v3B adapter (physical adapter SHA confirmed) | adapter_native | adapter_serializer |

All four arms receive identical questions, local search/read corpus, tool interfaces, prompt except the adapter/formatter treatment, max steps=12, output-token limit=512, timeout, quantization, seed/environment. The extra formatter call is charged in tokens/elapsed time. Never use publisher answers to build prompts or retrieve evidence. The serializer can only extract a verbatim span from a completed answer and must fail closed otherwise. There are NO stronger inference-time models.

## Outcomes

Primary: strict end-to-end answer correctness; matched wrong→correct and correct→wrong, by-family task rates. Secondary: completion, evidence/support coverage, calls, total tokens, run-time, tool errors, 3 independent A/A repetitions/variance before a promotion decision. No undefined "semantic" metric: require an explicit annotation rubric and agreement if subsequently added.

Two-factor interaction: (adapter_serializer − adapter_native) − (base_serializer − base_native) on matched task scores. The CPU-only \`research/agent_factorial_gate.py\` accepts pre-frozen task-level 2×2 records, validates source and model identifiers, computes paired transitions and descriptive interaction, and **never makes a promotion claim**.

Pilot n=24 across at least two families tests feasibility only, not high-powered statistical significance. A subsequent source-independent preregistered replication is required before claiming generalizable adapter gains. No GAIA2023 validation case may be labeled untouched: all 165 have known historical exposure.

## Decision

- If adapter improves native end-to-end success with independent-family replication and acceptable regression/cost, **KEEP Hweights for further testing**.
- If only formatter improves completed public QA tasks, **KEEP Hserialization only for that capability**; do not infer GAIA or QLoRA gain.
- If all effects vanish or mixed-source repeats are unstable, **KILL promotion, not entire hypothesis class**.
- Until physical adapter/source manifests and model-run budget are available: **BLOCKED; no new GPU, no unapproved training, no modifications to production agent defaults**.

Relevant existing negative GAIA serializer/grammar runs: 44013989299 and 44019807021. Positive public 2Wiki48 is 44022276727, but does not establish weight gains.
