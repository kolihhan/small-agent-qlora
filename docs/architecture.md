# Architecture

## Runtime

The project keeps a minimal autonomous loop: Qwen3.5-4B chooses a tool call or a final answer; the harness executes tools, returns bounded observations, blocks exact duplicates, and records visible traces. There is no planner/router/critic graph or GAIA-specific routing policy.

The Transformers diagnostic path performs one real pre-inference smoke of the
exact default tool surface (`search`, `read`, `inspect`, `python`) and fails
closed on a missing dependency or broken tool. The local Qwen path uses the
native default KV cache; no extra planner, parser framework, or compatibility
layer was added.

## Local evaluation protocol

The pinned GAIA validation pool is SHA-256 ranked within each level using one fixed local seed. The first 8/13/4 rows form the 25-task diagnostic partition. The next 32/52/16 rows form the 100-task evaluation partition. They are deterministic and disjoint; the remaining 40 tasks are unused by the current protocol.

Diagnostic tasks may be inspected repeatedly to classify observed failures and decide whether policy adaptation is a sensible intervention. The evaluation partition is frozen from this protocol onward and is used only after the adapter/config is frozen.

The runner records known semantic image/audio/video capability gaps separately from supported tasks. Rule-based failure labels are diagnostic proxies, not causal diagnoses.

## Training boundary

GAIA is evaluation-only. A local protection artifact stores normalized SHA-256 hashes of GAIA validation questions. Collection/training reject sources containing `gaia` and exact normalized protected-question overlaps. This catches direct copying, not semantic paraphrases.

QLoRA consumes only verified non-GAIA trajectories. LoRA is applied to the language backbone while the same agent loop and tool surface remain unchanged.

## Controlled comparison

The final comparator requires identical task IDs and identical recorded run configuration except for the adapter path. It reports Base/+LoRA accuracy, paired recoveries/regressions, efficiency metrics, and the same transitions restricted to capability-supported tasks.

The frozen P4 run never reached this comparison. Both tested cache placements
failed the sealed host-RAM floor on this machine, and the final native-cache
Diagnostic25 attempt stopped after 13 persisted records. The training-data
gate was independently false, so no adapter or evaluation100 run was allowed.
