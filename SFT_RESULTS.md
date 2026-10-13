# SFT / QLoRA Evidence

This project includes a real supervised fine-tuning experiment on a tool-using Qwen3.5-9B agent. The goal was not to manufacture a benchmark win; it was to test whether a small, controlled SFT intervention could change tool-use behavior under a frozen evaluation setup.

## Result

| Arm | GAIA Evaluation100 exact match | Completed | Tool calls | Tool errors | Duplicate blocks |
|---|---:|---:|---:|---:|---:|
| Base Qwen3.5-9B | 12 / 100 | 41 / 100 | 781 | 45 | 69 |
| QLoRA v3B | **15 / 100** | **47 / 100** | 777 | **35** | **52** |

The paired comparison contained **7 QLoRA-only wins vs 4 Base-only wins**. The +3 exact-match gain was **not statistically significant** (McNemar p = 0.5488), so I treat it as evidence that the SFT pipeline worked end-to-end and changed behavior, not as proof of a general benchmark improvement.

## What was fine-tuned

- Base model: Qwen3.5-9B
- Method: QLoRA / PEFT
- Objective: completion-only supervised fine-tuning on native Qwen tool-call chat formatting
- LoRA config held fixed for this probe: rank 16, alpha 32, dropout 0.05, LR 1e-4, all-linear target modules
- Runtime was not redesigned for the SFT comparison

## Training data

The synthetic policy dataset was generated independently of GAIA:

- train / dev / test: **1600 / 200 / 200** tasks
- eight policy capabilities: tool selection, argument grounding, multi-step execution, evidence-to-final, no-tool stopping, failure recovery, strategy switching, duplicate avoidance
- v3B selected a **64 supervised assistant-turn exposure** weighted toward diagnostic failure modes
- GAIA questions, answers, traces, entities, and case-specific routes were excluded from training
- protected-question hash checks and provenance checks fail closed on forbidden overlap

## Evaluation controls

The Base and QLoRA arms used the same frozen Evaluation100 selection, model family, tool surface, scorer, 16k context, non-thinking inference mode, and runtime contract. The adapter artifact was checksum-pinned before evaluation.

This matters because the comparison is intended to isolate the adapter treatment rather than mix SFT with unrelated runtime changes.

## Reproducible evidence

- Base Evaluation100 run: https://github.com/kolihhan/small-agent-qlora/actions/runs/38326300262
- QLoRA v3B Evaluation100 run: https://github.com/kolihhan/small-agent-qlora/actions/runs/38337206077
- Paired audit run: https://github.com/kolihhan/small-agent-qlora/actions/runs/38342561074
- v3B training plan: `docs/superpowers/plans/2026-10-11-policy-v3b-training.md`
- policy generator: `src/gaia_small_agent/training/oracle_policy_v3.py`
- QLoRA pipeline: `src/gaia_small_agent/training/qlora.py`

## What this demonstrates

The important result is not “15% is a great GAIA score.” It is that I built and ran the full SFT loop: dataset design and split isolation, trajectory verification, leakage protection, QLoRA training, adapter conversion/checksumming, frozen matched evaluation, and paired error analysis — then reported the result without hiding the negative statistical test.
