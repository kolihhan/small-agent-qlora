# TinyAgent-style QLoRA Policy Design

## Goal

Fine-tune `Qwen/Qwen3.5-4B` with QLoRA so the local Small Agent learns a better **tool-use policy** on unseen tasks. The training target is not world knowledge; it is the visible decision boundary:

`current conversation state + four tool schemas -> next assistant action`

The next action is one of `search`, `read`, `inspect`, `python`, or a final answer.

## Scope

Keep the existing single-agent runtime, four tools, Qwen chat/tool format, QLoRA trainer, GAIA protocol, and stable-search fix. Do not add ToolRAG, LLMCompiler, planner graphs, multi-agent orchestration, vector databases, or a teacher model in v1.

This branch must not change agent runtime behavior. It changes only training curriculum/evaluation plumbing.

## Reference pattern

Follow the useful part of TinyAgent: curate high-quality function/tool-calling supervision for a small model and train it with LoRA-style adaptation. All four tools remain visible as distractors so the model has to learn tool selection rather than being handed the correct tool.

## QLoRA target capabilities

V1 trains five balanced capabilities:

1. `tool_selection` — choose the correct tool while all four tools are available.
2. `argument_grounding` — emit the correct tool arguments from the user request/state.
3. `multi_step` — choose the next tool after a prior tool observation.
4. `evidence_to_final` — stop calling tools once evidence is sufficient and answer from the observation.
5. `no_tool_stop` — answer directly when no tool is needed; avoid unnecessary calls.

Error-recovery trajectories are excluded from v1 so the existing clean-trajectory contract remains unchanged.

## Dataset

Generate a deterministic, GAIA-independent synthetic oracle curriculum with 2,000 rows total:

- 1,600 train
- 200 dev
- 200 held-out test
- balanced within each split across the five capabilities

Each row carries complete provenance and an explicit split. Train/dev/test use independent deterministic seed namespaces. No GAIA text, entities, task IDs, answers, or traces may be used.

Oracle trajectories contain only visible messages: system, user, assistant tool calls, tool observations, and assistant final answers. They contain no hidden chain-of-thought.

Each row exposes the real four Small Agent tool schemas. Tool-call messages use the same OpenAI/Qwen-compatible structure already produced by `trajectory_from_result`.

## Verification contract

Oracle rows are accepted for training only when the generator can prove by construction that:

- the final answer equals the deterministic oracle;
- required tool calls are present;
- tool arguments are exactly the generated oracle arguments;
- there are no tool errors or duplicate calls;
- provenance is complete;
- source is non-GAIA.

Existing `load_verified_rows` remains the training gate.

## Training

Keep the current QLoRA hyperparameters unchanged for v1 so the curriculum is the only training variable:

- 4-bit NF4 + double quantization
- LoRA rank 16
- alpha 32
- dropout 0.05
- learning rate `1e-4`
- 2 epochs
- completion-only loss
- text-only causal language model

## Held-out evaluation

Before consuming another blind GAIA Shadow exposure, compare Base Qwen and Base+QLoRA on the 200 held-out policy tasks using the same runtime/tool surface.

Primary metric: end-to-end exact success rate.

Secondary metrics: completion rate, required-tool satisfaction, tool-call count, tool errors, duplicate blocks.

Promotion gate for QLoRA v1: **at least +10 percentage points exact success** on the held-out 200 with no increase in catastrophic model/runtime failures. If it does not pass, do not spend a Shadow exposure on this adapter.

## Final scientific comparison

If the adapter passes the held-out gate, freeze it and evaluate under the existing blind protocol. The final research comparison remains:

- A: Original Base
- B: Best stable policy/runtime
- C: B + QLoRA adapter

`C - B` is the QLoRA contribution.
