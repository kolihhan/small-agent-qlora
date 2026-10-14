# Held-out SFT result — Qwen3.5-9B QLoRA v3B

This is the primary direct-learning result for the v3B adapter.

## Result

| Metric | Base | QLoRA v3B | Delta |
|---|---:|---:|---:|
| Held-out next-action exact match | 20/64 (31.25%) | 32/64 (50.00%) | **+18.75 pp** |
| Tool-call decisions | 5/35 (14.29%) | 18/35 (51.43%) | **+37.14 pp** |
| Final-answer decisions | 15/29 (51.72%) | 14/29 (48.28%) | -3.45 pp |

Paired transitions: **16 v3B-only wins**, **4 Base-only wins**, 16 both-correct, 28 both-wrong.
Two-sided exact McNemar test over the 20 discordant pairs: **p = 0.0118**.

## Capability breakdown

| Capability | Base | v3B |
|---|---:|---:|
| Tool selection | 3/6 | **6/6** |
| Multi-step | 4/12 | **8/12** |
| Evidence → final | 5/20 | **9/20** |
| Strategy switch | 0/6 | **1/6** |
| Argument grounding | 2/6 | 2/6 |
| Failure recovery | 0/8 | 0/8 |
| Duplicate avoidance | 4/4 | 4/4 |
| No-tool stop | 2/2 | 2/2 |

The largest behavioral change is action control: `wrong_action_type` falls from **10 to 2**.
The remaining bottlenecks are argument grounding and final synthesis/recovery.

## Protocol

- Frozen `synthetic-small-agent-policy-v3/test` split; train/dev/test use distinct prompt families.
- 64 deterministic held-out assistant decisions selected before scoring with seed `heldout-v3b-next-action-v1`.
- Same Qwen3.5-9B Q4_K_M base, llama.cpp runtime, tool schemas, context, and decoding for both arms.
- Only treatment difference: frozen v3B GGUF LoRA adapter.
- Thinking mode enabled with reasoning budget 256; temperature 0 for a deterministic paired comparison.
- Teacher-forced immediate next-action evaluation; tool calls require exact tool + JSON arguments, final answers use the existing exact-answer scorer.
- GitHub Actions CPU only for inference. No ZeroGPU compute was used for this evaluation.

Source workflow runs: `38423661192` (shards 1–5,7) and `38426519870` (recovery shards 0,6).

## Interpretation

This result supports a narrow claim: **the v3B SFT adapter learned a better held-out tool-use policy**, especially tool selection and multi-step action decisions.

It does **not** by itself prove an end-to-end benchmark breakthrough. GAIA remains external generalization evidence, and the next engineering target is improving argument grounding, failure recovery, and final-answer synthesis without losing the tool-policy gain.
