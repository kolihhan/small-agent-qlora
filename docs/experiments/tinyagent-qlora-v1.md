# TinyAgent-style QLoRA v1 — Experiment Record

Status: **in progress; Base held-out complete, QLoRA training compute validation in progress**

## Frozen question

Can TinyAgent-style tool-policy QLoRA improve unseen tool-use performance of `Qwen3.5-4B` beyond the same unfine-tuned Base runtime?

The QLoRA target is the visible next-action policy:

`conversation state + all four tool schemas -> search/read/inspect/python/final answer`

No GAIA examples, entities, answers, traces, or per-case Shadow information are used for training.

## Frozen curriculum

Seed: `tinyagent-policy-v1`

Dataset:

- train: 1,600 rows
- dev: 200 rows
- held-out test: 200 rows
- five balanced capabilities: `tool_selection`, `argument_grounding`, `multi_step`, `evidence_to_final`, `no_tool_stop`
- all four Small Agent tool schemas are visible on every row
- deterministic oracle trajectories only

QLoRA configuration remains:

- model: `Qwen/Qwen3.5-4B`
- 4-bit NF4 + double quantization
- LoRA rank 16
- alpha 32
- dropout 0.05
- LR `1e-4`
- 2 epochs
- completion-only loss
- max length 1536

## Frozen Base held-out result

Workflow run: `37956168028`

Runtime model: `qwen3.5:4b`

Ollama model id: `d8b0f5e9760c`

Held-out aggregate:

- total: **200**
- exact correct: **153 / 200**
- exact success: **76.5%**
- completed: **200 / 200 (100%)**
- required-tool satisfaction: **200 / 200**
- tool calls: **200**
- tool errors: **0**
- duplicate calls blocked: **0**
- stop reasons: `final=200`

Aggregate exact correctness by frozen capability:

- `tool_selection`: **16 / 40**
- `argument_grounding`: **17 / 40**
- `multi_step`: **40 / 40**
- `evidence_to_final`: **40 / 40**
- `no_tool_stop`: **40 / 40**

Interpretation boundary: because required-tool satisfaction is 200/200, these aggregate results do **not** prove that the lower-scoring categories failed specifically from wrong tool choice. Case-level held-out outputs remain unopened.

## QLoRA promotion gate

The adapter must reach at least:

**86.5% held-out exact success**

That is the preregistered `+10.0 percentage point` threshold over Base. It must also introduce no new catastrophic model/runtime failures.

If it does not pass this held-out gate, v1 does not consume another GAIA Shadow exposure.

## Training compute validation

CPU smoke attempt 1 used an invalid `max_length=512`; the tool schemas consumed the prompt budget and TRL discarded fully masked completions. This was a smoke configuration error, not a model result.

CPU smoke attempt 2 restored the production `max_length=1536`. Dataset preprocessing succeeded and the Qwen3.5-4B 4-bit model loaded, but TRL 1.15 invoked a Triton-only fused LM-head loss path on a CPU runner and failed with `0 active drivers`.

The CPU-only smoke path was then isolated to use the model's native causal-LM loss while preserving the same completion-only labels. GPU training keeps the standard TRL fused path. The regression tests and normal CI are green on Ubuntu and Windows.

A one-row CPU smoke is used only to prove that a real QLoRA update can complete; it is not an experiment result and does not alter the frozen 1,600-row training configuration.
