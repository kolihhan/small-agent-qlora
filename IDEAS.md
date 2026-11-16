# Ranked ideas — GAIA / QLoRA — 2026-11-16

## Selected M0: scorer parity + exposure/data gate (no model calls)

- **Hypothesis:** without matching official scoring, frozen IDs, repeated seeds and decontamination we cannot identify real GAIA gain.
- **Predeclared pass:** scripted scorer parity against approved reference edge cases, reproducible Level-based DEV manifest, per-cohort exposure audit, and audited GAIA+BIRD DEV exclusion corpora. Missing full references must yield **BLOCKED**, not silent pass.
- **Estimated cost:** 0 GPU-hours; 0 model/tool calls; CPU-only tests. No score can be promoted.
- **Status:** not run as of creation; not a model improvement.

## M1: hard agent failures on external source-disjoint tasks

Diagnose actual evidence-path visits, delivered tokens and final-answer format on new non-GAIA/non-exposed benchmark families. Register n, budget and one mechanism, then Base vs scaffold change paired ≥3 seeds. No reuse of old 24 MuSiQue holdouts. Do not call changed answer formatting semantic correctness.

## M2: constrained tool schema

Target tool-call format / retries only if M1 finds a substantial measurable cause. JSON-schema constrained decoding versus identical unmodified runtime, paired tasks, report step/token/tool cost, no baseline model change.

## M3: QLoRA rejection-sampled trajectories

Use only verified-correct non-GAIA, non-BIRD DEV trajectories from new task families. Prioritize longer-horizon evidence selection/recovery and stop decisions with sound provenance. Training grid proposed only after M1 and source audit. 2×2 factorial: prompt × adapter, locked task IDs/seeds/tools/budgets, independent family evidence and general regression suite. No stronger teacher without explicit prior authorization.
