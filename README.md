<h1 align="center">Small-Agent QLoRA</h1>

<p align="center">
  <strong>Diagnose agent failures → design SFT data → run QLoRA → measure whether the agent actually gets better.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-111827?style=flat-square&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Qwen3.5--9B-111827?style=flat-square" alt="Qwen3.5-9B" />
  <img src="https://img.shields.io/badge/QLoRA-111827?style=flat-square" alt="QLoRA" />
  <img src="https://img.shields.io/badge/llama.cpp-111827?style=flat-square" alt="llama.cpp" />
  <img src="https://img.shields.io/badge/GitHub_Actions-111827?style=flat-square&logo=githubactions" alt="GitHub Actions" />
</p>

<p align="center">
  <a href="#result">Result</a> ·
  <a href="#external-gaia-check">External GAIA</a> ·
  <a href="#what-changed">What changed</a> ·
  <a href="#runtime">Runtime</a> ·
  <a href="#experiment-design">Experiment</a> ·
  <a href="#reproduce">Reproduce</a> ·
  <a href="docs/v3d-results.md">Full evidence</a>
</p>

## At a glance

| | |
|---|---|
| **Question** | Can targeted post-training improve a small tool agent's actual closed-loop behavior, not just its training loss? |
| **Model used for the final experiment** | Qwen3.5-9B, evaluated as Q4_K_M with `llama.cpp` |
| **Agent** | One model chooses a tool call or a final answer; no planner / router / critic graph |
| **Training** | QLoRA / SFT on synthetic, independently generated policy trajectories |
| **Final result** | **34.5% Base → 52.5% v3B → 62.0% v3D** on a frozen 200-task deterministic closed-loop test |
| **External check** | Matched GAIA100: **9% Base → 18% v3B → 18% v3D**; v3D exposed an over-search / finalization regression |
| **Main lesson** | More trajectory supervision was not automatically better. State-aware corrective data worked better than the full-trajectory ablation, but the external benchmark exposed a stopping-policy weakness. |

The project is intentionally evidence-first: negative experiments stay in the record, promotion rules are fixed before the result, and the headline claim is bounded to what the evaluation actually measures.

## Result

The final matched closed-loop comparison used the same runtime commit, frozen 200-task test split, deterministic fixture-backed tools, temperature 0, reasoning budget 0, max 12 agent steps, and max 512 new tokens per call.

| Arm | Correct | Exact success | Completion |
|---|---:|---:|---:|
| Base | 69 / 200 | 34.5% | 98.0% |
| v3B | 105 / 200 | 52.5% | 100.0% |
| **v3D** | **124 / 200** | **62.0%** | **99.5%** |

**v3D vs v3B: +19 correct tasks / +9.5 percentage points.**

The preregistered promotion rule required v3D to gain at least 5 correct tasks over v3B while losing no more than 2 correct tasks in any capability bucket. v3D passed with **+19 total correct** and a **worst capability delta of 0**.

Evidence: GitHub Actions run [`38752732871`](https://github.com/kolihhan/small-agent-qlora/actions/runs/38752732871), artifact `closed-loop-v3d-summary`.

### Where the gain came from

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

So the supported claim is not “v3D is better at everything.” The measured improvement is concentrated in **stopping correctly when a tool is unnecessary**, plus a smaller strategy-switch gain. Failure recovery remains unsolved.

See [`docs/v3d-results.md`](docs/v3d-results.md) for the controlled experiment evidence and caveats.

## External GAIA check

The final adapters were also run on a frozen, matched **GAIA 2023 validation Evaluation100** check. Each five-task shard ran Base → v3B → v3D sequentially on the same GitHub runner with the same evaluator, temperature 0, reasoning budget 0, and a 12-step ceiling.

| Arm | Correct | Accuracy | Completion | `invalid_final` | Duplicate calls blocked |
|---|---:|---:|---:|---:|---:|
| Base | 9 / 100 | 9% | 45% | 42 | 78 |
| v3B | 18 / 100 | 18% | 44% | 51 | 66 |
| v3D | 18 / 100 | 18% | 25% | 69 | 127 |

The external result supports a v3B gain over Base, but **v3D did not improve GAIA accuracy over v3B**. Paired trace analysis showed a stopping regression instead: v3D more often kept calling tools until the 12-step ceiling, with 20 v3B-final → v3D-non-final transitions versus 1 in the opposite direction. Tool errors did not increase.

This does not replace the primary controlled result. It narrows the conclusion: v3D improved the targeted deterministic policy benchmark, while GAIA exposed an **over-search / failure-to-finalize** weakness on longer external trajectories.

Evidence: GitHub Actions run [`38911211243`](https://github.com/kolihhan/small-agent-qlora/actions/runs/38911211243), artifact `matched-gaia-v3d-summary`. See [`docs/gaia-v3d-results.md`](docs/gaia-v3d-results.md) for the paired failure analysis.

## What changed

The useful part of this project was not just running QLoRA. It was diagnosing why different training-data choices changed behavior.

### v3B — strong first post-training baseline

The frozen 64 next-action probe improved from:

- Base: **20 / 64 (31.25%)**
- v3B: **32 / 64 (50.0%)**

v3B sharply improved tool-use decisions, but recovery and some finalization failures remained.

### v3C — negative full-trajectory ablation

The tempting next move was to supervise fuller trajectories so the model could see more recovery and multi-step context.

That did **not** work overall:

- v3B: **32 / 64**
- v3C: **27 / 64**
- v3C finals: **8 / 29** vs v3B **14 / 29**
- v3C `wrong_final`: **21**

The full-trajectory treatment improved a few recovery decisions but damaged finalization enough to fail the preregistered gate.

Evidence: GitHub Actions run [`38497917564`](https://github.com/kolihhan/small-agent-qlora/actions/runs/38497917564).

### v3D — state-aware corrective continuation

The root data-design bug was that a capability label alone did not tell us **which state inside a trajectory** should be supervised. In recovery / strategy-switch examples, the first action was deliberately wrong; selecting by capability + action type could accidentally train that failure as the target behavior.

v3D therefore selected 24 state-aware corrective decisions:

- 12 tool calls + 12 final actions
- task-unique
- excludes v3B exposure task IDs
- explicitly targets the corrective / later trajectory state
- excludes deliberately failing first actions in recovery and strategy-switch trajectories

The one-step probe did **not** show an aggregate win:

| Arm | Frozen next-action exact |
|---|---:|
| Base | 20 / 64 (31.25%) |
| v3B | 32 / 64 (50.0%) |
| v3D | 32 / 64 (50.0%) |

But the internal capability mix changed: multi-step and strategy-switch improved while evidence-to-final regressed. That is why the project did not stop at one-step scoring and moved to actual closed-loop evaluation.

Evidence: GitHub Actions run [`38714014392`](https://github.com/kolihhan/small-agent-qlora/actions/runs/38714014392).

## Runtime

The runtime stays deliberately small:

```text
user task
   ↓
Qwen model
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

The harness owns the execution boundaries around the model:

- explicit model timeout / unavailable / error states;
- tool exceptions become bounded observations instead of crashing the run;
- exact duplicate tool calls are blocked;
- URL reads block local/private destinations and stream within a source-size budget;
- workspace files are size-checked before expensive parsing;
- Python runs in an isolated, time-limited subprocess with a restricted builtin surface;
- `small-agent doctor` checks backend readiness;
- `small-agent run --output ...` persists answer, stop reason, metrics, trace, and effective runtime configuration.

This is a production-style local harness, not a hardened multi-tenant sandbox.

## Experiment design

The final evidence ladder separates local policy diagnosis from actual end-to-end behavior:

```text
synthetic policy train split
        ↓
trajectory / state diagnosis
        ↓
QLoRA treatment
        ↓
frozen next-action probe
        ↓
deterministic closed-loop test
        ↓
external benchmark evidence (secondary)
```

The primary controlled result is the deterministic non-GAIA policy test because it gives a stable tool environment and lets Base / v3B / v3D run under the same harness.

GAIA is retained as external secondary evidence, not used to inflate the headline. The matched GAIA100 check showed **9% Base → 18% v3B → 18% v3D**: v3B generalized beyond the synthetic policy test, while v3D added no further external accuracy and exposed an over-search / finalization regression.

## Training

### v3B

v3B is the main earlier QLoRA checkpoint and the baseline for the corrective continuation. Its training used completion-only next-action supervision on selected policy decisions.

### v3D

v3D continues from v3B with:

- 24 corrective examples
- 24 optimizer steps
- learning rate `1e-4`
- max length 1024
- four sequential batches of six examples
- optimizer state preserved between batches
- **layer-31 LoRA parameters only trainable**

Training run: [`38561069365`](https://github.com/kolihhan/small-agent-qlora/actions/runs/38561069365).

The restricted trainable scope is a compute compromise, not a claim that layer-31-only continuation is equivalent to the original all-linear QLoRA setup.

### Why only layer 31 on CPU?

Standard GitHub-hosted CPU runners could load the 9B model, but the Qwen3.5 recurrent kernels fell back to slow reference PyTorch implementations.

Measured feasibility:

- all-layer CPU backward: still incomplete after ~5h51m, then cancelled;
- shortest 754-token layer-31-only optimizer step: ~24.1 minutes;
- `flash-linear-attention` Triton kernel: unavailable on the CPU runner (`0 active drivers`).

That evidence led to the bounded layer-31 continuation instead of pretending an all-layer CPU run was practical.

## Reproduce

Key workflows:

- [`.github/workflows/v3d-cpu-batched-continuation.yml`](.github/workflows/v3d-cpu-batched-continuation.yml) — sequential CPU corrective QLoRA
- [`.github/workflows/v3d-heldout-next-action.yml`](.github/workflows/v3d-heldout-next-action.yml) — frozen 64 next-action comparison
- [`.github/workflows/v3d-closed-loop.yml`](.github/workflows/v3d-closed-loop.yml) — matched 200-task Base / v3B / v3D closed-loop comparison
- [`.github/workflows/gaia-v3d-clean-same-runner.yml`](.github/workflows/gaia-v3d-clean-same-runner.yml) — matched external GAIA100 Base / v3B / v3D check

Key implementation files:

- [`src/gaia_small_agent/training/exposure_v3d.py`](src/gaia_small_agent/training/exposure_v3d.py) — state-aware corrective selector
- [`src/gaia_small_agent/training/cpu_v3d_batches.py`](src/gaia_small_agent/training/cpu_v3d_batches.py) — batch continuity / partition helpers
- [`scripts/run_v3d_cpu_batch.py`](scripts/run_v3d_cpu_batch.py) — resumable CPU training step
- [`src/gaia_small_agent/training/policy_eval.py`](src/gaia_small_agent/training/policy_eval.py) — deterministic closed-loop evaluator

### Local runtime

For a lightweight local smoke test, install the runtime extras:

```powershell
python -m pip install -e ".[search,files]"
small-agent doctor
small-agent run "Find the relevant evidence and answer concisely." `
  --output runs/task/result.json
```

The local runtime can use a smaller local model for development; **the reported post-training experiment above uses Qwen3.5-9B**.

### Evaluation / training dependencies

```powershell
python -m pip install -e ".[eval,search,files]"
```

The full Transformers / QLoRA path requires the heavier training extras and a compatible environment:

```powershell
python -m pip install -e ".[eval,search,files,train]"
small-agent doctor --backend transformers
```

## What gets recorded

A normal `small-agent run --output` artifact records:

- effective runtime configuration;
- final answer, completion flag, and stop reason;
- step / tool metrics;
- visible action trace.

Experiment workflows additionally preserve frozen selection metadata, model / adapter provenance, checksums, per-shard summaries, and aggregate comparison artifacts.

## Claim boundary

This repository supports the claim that:

> **A state-aware corrective QLoRA continuation improved exact success on the frozen deterministic 200-task closed-loop policy test from 52.5% (v3B) to 62.0%, after a negative full-trajectory ablation informed the data redesign.**

It also records matched external evidence showing Base **9%**, v3B **18%**, and v3D **18%** on GAIA100. That supports the v3B-over-Base generalization result, but it does **not** establish an additional external accuracy gain from v3D.

It does **not** establish that:

- v3D is universally better across every capability;
- failure recovery is solved;
- the synthetic deterministic test is a general real-world agent benchmark;
- layer-31-only CPU continuation is equivalent to full all-layer QLoRA;
- v3D is better than v3B on external GAIA.

That distinction is intentional. The project is meant to show diagnosis, controlled post-training, reproducible evaluation, and honest failure analysis rather than a benchmark-only success story.

## Repository shape

```text
small-agent-qlora/
├── src/gaia_small_agent/
│   ├── agent/
│   ├── model/
│   ├── tools/
│   ├── benchmark/
│   └── training/
├── scripts/
├── docs/
└── tests/
```

## Limits

- The Python tool is process-isolated and time-limited, but not a hardened hostile-code sandbox.
- `inspect` does not provide general semantic image / audio / video understanding.
- Search can use live web data in normal runtime operation, so exact external results can change.
- Runtime source-size limits reduce accidental memory blow-ups; they are not a complete security boundary.
- Exact hash leakage checks do not detect paraphrased benchmark questions.
- The synthetic policy curriculum is intentionally bounded and does not cover every autonomous-agent failure mode.
- `failure_recovery` remains a clear unresolved weakness in the final closed-loop result.
- v3D's GAIA100 trace shows a separate over-search / finalization weakness on longer external trajectories.

## Verification

CPU-only CI verifies package installation, CLI entry-point loading, tests, and compilation on Linux and Windows. Experiment success is verified separately by the frozen Actions workflows and their artifacts; ordinary CI passing is not treated as benchmark evidence.

See [`docs/verification.md`](docs/verification.md), [`docs/v3d-results.md`](docs/v3d-results.md), and [`docs/gaia-v3d-results.md`](docs/gaia-v3d-results.md).

## References

- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- [GAIA scorer](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
