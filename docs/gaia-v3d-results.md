# Matched GAIA100: Base vs v3B vs v3D

This note records the external GAIA check for the final Qwen3.5-9B post-training experiment. It is secondary evidence; the primary controlled result remains the frozen 200-task deterministic policy evaluation.

## Result

GAIA 2023 validation / frozen Evaluation100 was run with the same Runtime v2 evaluator for all three arms. Each five-task shard ran Base → v3B → v3D sequentially on the same GitHub runner.

Protocol:

- Qwen3.5-9B Q4_K_M
- runtime commit `08fbcaa5ed82c5238188c45d5748993bae9bf1e1`
- 100 matched GAIA tasks
- 20 shards × 5 tasks
- context 16,384 tokens
- temperature 0
- reasoning budget 0
- max 12 agent steps
- max 512 new tokens per call

| Arm | Correct | Accuracy | Completion | `invalid_final` | Duplicate calls blocked | Tool errors |
|---|---:|---:|---:|---:|---:|---:|
| Base | 9 / 100 | 9% | 45% | 42 | 78 | 39 |
| v3B | 18 / 100 | 18% | 44% | 51 | 66 | 37 |
| v3D | 18 / 100 | 18% | 25% | 69 | 127 | 35 |

Evidence: GitHub Actions run `38911211243`, artifact `matched-gaia-v3d-summary`.

The external result supports a v3B gain over Base (**9% → 18%**), but it does **not** show a further GAIA accuracy gain from v3D (**18% → 18%**).

## Paired failure analysis

The task-level v3B → v3D comparison was:

- correctness: 13 both correct, 5 v3B-only correct, 5 v3D-only correct, 77 both wrong;
- finalization: 24 final→final, **20 v3B final→v3D non-final**, 1 non-final→final, 55 non-final→non-final;
- the 20-vs-1 finalization imbalance gives a two-sided exact McNemar `p ≈ 2.1e-5`.

Every v3D `invalid_final` occurred at step 12 while the model was still attempting another tool call. v3D also made 876 tool calls versus 784 for v3B and triggered 127 duplicate-call blocks versus 66. Tool errors did not increase; they fell slightly from 37 to 35.

A useful diagnostic, though **not** an official benchmark metric, is correctness conditional on actually emitting a final answer:

- Base: 9 / 45 = 20.0%
- v3B: 18 / 44 = 40.9%
- v3D: 18 / 25 = 72.0%

So the dominant external failure mode is **termination / over-search**, not a generic loss of tool competence. When v3D does finalize, its answers are stronger; on GAIA it too often keeps searching until the 12-step ceiling.

Four of the five v3B-correct → v3D-wrong transitions were max-step non-finalization failures. The fifth was an answer-format regression: v3B returned the exact answer `519`, while v3D returned a verbose explanation ending in `519` and failed exact scoring.

## Interpretation

This result contrasts with the deterministic policy test, where v3D's largest gain was `no_tool_stop` (**6/25 → 23/25**) and total exact success improved **52.5% → 62.0%** over v3B.

The evidence therefore supports a bounded interpretation:

- v3D improved the targeted synthetic policy behavior substantially;
- v3B's post-training improvement generalized to GAIA versus Base;
- v3D did not add further GAIA accuracy over v3B;
- v3D developed a GAIA-specific stopping regression: it searched more, duplicated more calls, and finalized much less often.

The 24-example v3D continuation used 12 tool-call targets and 12 final targets. The selected final targets occur only at assistant-message indices 2, 4, or 6, so a plausible explanation is short-horizon synthetic stopping supervision failing to transfer to GAIA's longer search trajectories. The layer-31-only continuation may also contribute, but this experiment does not isolate that factor causally.

## Decision

Do **not** start a v3E solely to chase this GAIA100 subset.

Increasing the step budget or forcing a last-step answer would patch the runtime rather than demonstrate better post-training. Another continuation would require a new preregistered experiment and risks tuning to the external evaluation set.

Keep v3D as the deterministic-policy champion, and report GAIA honestly as secondary external evidence:

> **Deterministic closed-loop:** Base 34.5% → v3B 52.5% → v3D 62.0%.  
> **Matched GAIA100:** Base 9% → v3B 18% → v3D 18%; v3D's main external regression is over-search / failure to finalize.
