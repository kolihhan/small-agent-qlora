# v3D result: state-aware corrective QLoRA

This note records the final evidence chain for the v3D experiment. The goal is to separate what the training change actually improved from what it did not.

## Result in one line

On the frozen 200-task deterministic closed-loop policy test, the final v3D adapter reached **124/200 exact success (62.0%)**, compared with **105/200 (52.5%)** for v3B and **69/200 (34.5%)** for the base model.

That is **+9.5 percentage points / +19 correct tasks over v3B** and **+27.5 percentage points / +55 correct tasks over Base** under the same runtime commit and evaluation settings.

The preregistered v3D promotion rule required at least +5 correct tasks over v3B and no capability losing more than 2 correct tasks. v3D passed: total gain was +19 and the worst capability delta was 0.

**Champion: v3D on the controlled deterministic policy evaluation.**

Evidence: GitHub Actions run `38752732871`, artifact `closed-loop-v3d-summary`.

## Frozen closed-loop comparison

Protocol:

- Qwen3.5-9B Q4_K_M base model
- same runtime commit for all arms: `4176358a5b9bf700ae8817e4348e6a9b3965e49c`
- 200 frozen policy-v3 test tasks
- 10 shards × 20 tasks
- deterministic fixture-backed tools
- temperature 0
- reasoning budget 0
- max 12 agent steps
- max 512 new tokens per model call

| Arm | Correct | Exact success | Completion | Required tools satisfied |
|---|---:|---:|---:|---:|
| Base | 69 / 200 | 34.5% | 98.0% | 147 / 200 |
| v3B | 105 / 200 | 52.5% | 100.0% | 175 / 200 |
| **v3D** | **124 / 200** | **62.0%** | **99.5%** | **175 / 200** |

### Where v3D improved over v3B

| Capability | v3B | v3D | Delta |
|---|---:|---:|---:|
| argument_grounding | 12 / 25 | 12 / 25 | 0 |
| duplicate_avoidance | 25 / 25 | 25 / 25 | 0 |
| evidence_to_final | 12 / 25 | 12 / 25 | 0 |
| failure_recovery | 0 / 25 | 0 / 25 | 0 |
| multi_step | 25 / 25 | 25 / 25 | 0 |
| **no_tool_stop** | **6 / 25** | **23 / 25** | **+17** |
| **strategy_switch** | **0 / 25** | **2 / 25** | **+2** |
| tool_selection | 25 / 25 | 25 / 25 | 0 |

The aggregate improvement is therefore concentrated, not universal. v3D mostly learned to stop and answer directly when no tool was needed, while adding a smaller strategy-switch gain. It did **not** solve failure recovery.

This matters because the headline is not "v3D became better at everything." The supported claim is narrower: **the corrective continuation changed agent behavior in a way that produced a large end-to-end gain on the held-out deterministic tool-policy test without regressing the other measured capability buckets.**

## Why the one-step evaluation did not predict the full gain

Before closed-loop evaluation, v3D was tested on the same frozen 64 next-action decisions used for Base and v3B:

| Arm | Exact next-action |
|---|---:|
| Base | 20 / 64 (31.25%) |
| v3B | 32 / 64 (50.0%) |
| v3D | 32 / 64 (50.0%) |

v3D tied v3B overall. The capability mix moved, however:

- multi_step: 8/12 → 10/12
- strategy_switch: 1/6 → 3/6
- evidence_to_final: 9/20 → 6/20
- failure_recovery: 0/8 → 0/8
- final actions overall: 14/29 → 14/29

Paired v3D vs v3B was 4 wins / 4 losses, exact McNemar p = 1.0.

Evidence: GitHub Actions run `38714014392`, artifact `heldout-v3d-thinking256-summary`.

The important lesson is that a one-step policy probe and a closed-loop agent evaluation measure different failure surfaces. The 64-decision probe was useful for diagnosis, but it was not sufficient to estimate compounded end-to-end behavior.

## Negative ablation: v3C

v3C tested the tempting idea that using fuller trajectories would teach recovery and multi-step behavior better. It did improve some tool-use decisions, but it damaged finalization badly.

Frozen 64 next-action result:

| Arm | Correct | Tool calls | Finals |
|---|---:|---:|---:|
| v3B | 32 / 64 | 18 / 35 | 14 / 29 |
| v3C | 27 / 64 | 19 / 35 | 8 / 29 |

v3C produced 21 `wrong_final` errors and failed the preregistered promotion bar. Against v3B it had only 2 exclusive wins versus 7 v3B-only wins.

Evidence: GitHub Actions run `38497917564`, artifact `heldout-v3c-thinking256-summary`.

This negative result changed the data design: instead of supervising whole trajectories indiscriminately, v3D selected state-aware corrective decisions and explicitly avoided labeling the deliberately failing first action in recovery / strategy-switch trajectories as the desired behavior.

## What v3D actually trained

v3D is a corrective continuation from the v3B adapter, not a fresh full-model re-train.

- 24 state-aware corrective examples
- 12 tool calls + 12 final actions
- task-unique selection
- v3B exposure task IDs excluded
- 24 optimizer steps
- learning rate `1e-4`
- max length 1024
- four sequential CPU batches of six examples
- optimizer state carried across batches
- **only layer-31 LoRA parameters trainable**

Training evidence: GitHub Actions run `38561069365`.

The restricted trainable scope is a computational compromise, not an equivalent replacement for the original all-linear v3B QLoRA setup. It must be kept in the interpretation of the experiment.

## Why CPU training was restricted

The project tested whether standard GitHub-hosted CPU runners could continue the 9B adapter without paid GPU compute.

Evidence from the feasibility work:

- an all-layer CPU backward step did not finish after roughly 5h51m and was cancelled;
- a shortest 754-token, layer-31-only optimizer step completed in about 24.1 minutes;
- the Qwen3.5 recurrent kernels fell back to reference PyTorch CPU implementations;
- `flash-linear-attention` could install but its Triton kernel could not run on the CPU runner (`0 active drivers` / unsupported platform).

The final v3D training therefore used the smallest trainable scope that was operationally feasible on standard GitHub CPU resources.

## Matched external GAIA100

The final adapters were also evaluated on **GAIA 2023 validation / frozen Evaluation100** with a matched same-runner protocol: each five-task shard ran Base → v3B → v3D sequentially under the same Runtime v2 evaluator.

| Arm | Correct | Accuracy | Completion | `invalid_final` | Duplicate calls blocked |
|---|---:|---:|---:|---:|---:|
| Base | 9 / 100 | 9% | 45% | 42 | 78 |
| v3B | 18 / 100 | 18% | 44% | 51 | 66 |
| v3D | 18 / 100 | 18% | 25% | 69 | 127 |

Evidence: GitHub Actions run `38911211243`, artifact `matched-gaia-v3d-summary`.

The external result supports a **v3B-over-Base gain (9% → 18%)**, but v3D added **no further GAIA accuracy gain (18% → 18%)**. Paired trace analysis identified a stopping regression: v3D had 20 v3B-final → v3D-non-final transitions versus 1 in the opposite direction, made more tool calls, and triggered substantially more duplicate-call blocks while tool errors did not increase.

The dominant external failure mode is therefore **over-search / failure to finalize**, not a generic loss of tool competence. This is deliberately kept as secondary evidence rather than used to inflate the controlled synthetic headline.

See `docs/gaia-v3d-results.md` for the full paired failure analysis and claim boundaries.

## Claim boundary

What this project can claim:

- an end-to-end SFT/QLoRA workflow was implemented and executed;
- failure analysis changed the training-data design;
- a negative full-trajectory ablation was measured and rejected;
- a state-aware corrective continuation was trained reproducibly under a constrained CPU setup;
- the resulting v3D adapter improved held-out deterministic closed-loop exact success from 52.5% (v3B) to 62.0%, with a preregistered promotion rule;
- under the matched GAIA100 external check, v3B improved over Base from 9% to 18%.

What it should **not** claim:

- that v3D is universally better than v3B on every capability;
- that failure recovery is solved (it remains 0/25 on the closed-loop test);
- that the layer-31 CPU continuation is equivalent to full all-layer QLoRA;
- that the synthetic deterministic benchmark is a general real-world agent benchmark;
- that v3D is better than v3B on external GAIA;
- that one 100-task GAIA subset establishes broad real-world generalization.

The evidence is therefore split cleanly: **v3D is the controlled deterministic-policy champion; v3B provides the observed external GAIA gain over Base; v3D's GAIA trace exposes a separate termination weakness.**

## Reproduce the evidence

Key workflows:

- `.github/workflows/v3d-cpu-batched-continuation.yml` — sequential CPU corrective training
- `.github/workflows/v3d-heldout-next-action.yml` — frozen 64 next-action comparison
- `.github/workflows/v3d-closed-loop.yml` — matched 200-task Base / v3B / v3D closed-loop comparison
- `.github/workflows/gaia-v3d-clean-same-runner.yml` — matched external GAIA100 Base / v3B / v3D comparison

Key implementation files:

- `src/gaia_small_agent/training/exposure_v3d.py` — state-aware corrective selector
- `src/gaia_small_agent/training/cpu_v3d_batches.py` — deterministic batch partition / continuity helpers
- `scripts/run_v3d_cpu_batch.py` — resumable CPU continuation
- `src/gaia_small_agent/training/policy_eval.py` — deterministic closed-loop evaluator

The Actions run IDs above preserve the actual executed configurations and artifacts; the README summarizes them, while this note keeps the experiment interpretation and caveats explicit.
