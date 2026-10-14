# SFT / QLoRA Results

This project tests whether supervised fine-tuning can teach a local Qwen3.5-9B agent a better tool-use policy, then checks whether that behavior transfers to a harder end-to-end agent benchmark.

## Headline result

On a **frozen 64-decision held-out tool-policy evaluation** drawn from the independent v3 test split, QLoRA improved next-action accuracy from **31.25% to 50.00%** under the same inference runtime.

| Arm | Correct | Next-action accuracy |
|---|---:|---:|
| Base Qwen3.5-9B | 20 / 64 | 31.25% |
| QLoRA v3B | **32 / 64** | **50.00%** |
| Delta | **+12 decisions** | **+18.75 pp** |

Paired transitions were **16 QLoRA-only wins vs 4 Base-only wins**, with 16 both-correct and 28 both-wrong cases. The exact paired McNemar/binomial test gives **p = 0.0118** for this frozen sample.

The largest effect was on the behavior the SFT curriculum directly targeted:

| Decision type | Base | QLoRA v3B | Delta |
|---|---:|---:|---:|
| Tool-call decisions | 5 / 35 (14.29%) | **18 / 35 (51.43%)** | **+37.14 pp** |
| Final-answer decisions | 15 / 29 (51.72%) | 14 / 29 (48.28%) | -3.45 pp |

The adapter also reduced `wrong_action_type` errors from **10 to 2**. This is the clearest evidence that the model learned the intended tool-policy behavior rather than merely changing its final-answer style.

## Capability breakdown

| Capability | Base | QLoRA v3B |
|---|---:|---:|
| Tool selection | 3 / 6 (50.0%) | **6 / 6 (100.0%)** |
| Multi-step | 4 / 12 (33.3%) | **8 / 12 (66.7%)** |
| Evidence → final | 5 / 20 (25.0%) | **9 / 20 (45.0%)** |
| Strategy switch | 0 / 6 (0.0%) | **1 / 6 (16.7%)** |
| Argument grounding | 2 / 6 (33.3%) | 2 / 6 (33.3%) |
| Failure recovery | 0 / 8 (0.0%) | 0 / 8 (0.0%) |
| Duplicate avoidance | 4 / 4 (100.0%) | 4 / 4 (100.0%) |
| No-tool stop | 2 / 2 (100.0%) | 2 / 2 (100.0%) |

This breakdown is useful because it also shows what the current SFT did **not** solve: failure recovery remains at 0%, argument grounding is unchanged, and final-answer decisions slightly regress. Those are the next curriculum targets rather than reasons to hide the result.

## Evaluation contract

This is a **held-out SFT learning evaluation**, not a GAIA leaderboard claim.

- Source: independent `test` split from the synthetic policy-v3 generator; train/dev/test use distinct prompt families.
- Frozen subset: 64 unique assistant decisions selected once with seed `heldout-v3b-next-action-v1` using the diagnostic-weighted capability/action mix.
- Evaluation mode: teacher-forced oracle prefix → predict the **immediate next assistant action**.
- Tool calls are scored on action type, tool name, and canonicalized arguments; final actions are exact-answer scored.
- Both arms use the same Qwen3.5-9B `Q4_K_M` GGUF, llama.cpp build, tool schema, context, temperature, and **reasoning budget 256**.
- The paired treatment difference is the v3B GGUF LoRA adapter.
- Compute: standard GitHub Actions CPU runners; no hosted inference endpoint was used for the evaluation.

Because both arms use thinking=256, this result measures the **adapter contribution under the current best inference setting**. It should not be mixed with the older thinking=0 GAIA comparison as if they were one experiment.

## External transfer: GAIA Evaluation100

The same v3B adapter was previously evaluated on the frozen GAIA Evaluation100 under a matched thinking=0 runtime:

| Arm | Exact match | Completed | Tool errors | Duplicate blocks |
|---|---:|---:|---:|---:|
| Base Qwen3.5-9B | 12 / 100 | 41 / 100 | 45 | 69 |
| QLoRA v3B | **15 / 100** | **47 / 100** | **35** | **52** |

That paired comparison had **7 QLoRA-only wins vs 4 Base-only wins**, but the +3 exact-match gain was **not statistically significant** (McNemar p = 0.5488). I therefore treat GAIA as directional external-transfer evidence, not as the primary SFT claim.

The combination of results is more informative than either number alone:

1. **Held-out policy eval:** the adapter clearly learns the supervised next-action behavior (+18.75 pp; p=0.0118).
2. **GAIA100:** some benefit transfers to a substantially harder end-to-end agent benchmark, but the transfer is still weak and statistically inconclusive.
3. **Error analysis:** the remaining gap is concentrated in recovery, grounding, strategy switching, and finalization rather than basic tool selection alone.

## What was fine-tuned

- Base model: Qwen3.5-9B
- Method: QLoRA / PEFT
- Objective: completion-only next-action SFT using native Qwen tool-call chat formatting
- LoRA: rank 16, alpha 32, dropout 0.05, learning rate 1e-4, all-linear target modules
- Policy-v3 data: **1600 / 200 / 200** train/dev/test tasks across eight capabilities
- v3B exposure: **64 supervised assistant decisions**, diagnostic-weighted, 35 tool calls + 29 finals
- Training rows require verified trajectories and complete provenance; GAIA-sourced rows and protected-question overlap are rejected

## Reproducible evidence

### Held-out next-action evaluation — thinking256

- Main CPU shard run (6 valid shards): https://github.com/kolihhan/small-agent-qlora/actions/runs/38423661192
- CPU recovery run for shards 0 and 6: https://github.com/kolihhan/small-agent-qlora/actions/runs/38426519870
- Evaluator implementation: `src/gaia_small_agent/training/next_action_eval.py` on `experiment/heldout-next-action-eval`

### Training / GAIA transfer

- v3B training artifact source: https://github.com/kolihhan/small-agent-qlora/actions/runs/38152689194
- Converted GGUF adapter: https://github.com/kolihhan/small-agent-qlora/actions/runs/38156235327
- Base Evaluation100: https://github.com/kolihhan/small-agent-qlora/actions/runs/38326300262
- QLoRA v3B Evaluation100: https://github.com/kolihhan/small-agent-qlora/actions/runs/38337206077
- Paired GAIA audit: https://github.com/kolihhan/small-agent-qlora/actions/runs/38342561074

## What this demonstrates

The result is not just that a LoRA adapter file was produced. The project covers the full post-training loop:

**failure analysis → independent curriculum design → split isolation and leakage guards → verified trajectories → QLoRA training → adapter conversion/checksumming → treatment-aligned held-out evaluation → paired statistical analysis → external transfer evaluation → bad-case analysis.**

The held-out result demonstrates that the SFT intervention changed the intended model policy in a measurable way. The GAIA result shows that stronger broad-agent transfer is still an open engineering problem, which the capability breakdown makes actionable rather than ambiguous.
