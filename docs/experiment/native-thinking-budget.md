# Native Thinking Budget Spike

Question: Does Qwen3.5-9B native bounded thinking improve agent behavior over thinking disabled, without changing the system prompt, tools, runtime policy, QLoRA weights, or GAIA task set?

Protocol:
- Base Qwen3.5-9B only.
- Keep the existing simple SYSTEM_PROMPT unchanged.
- Preserve native `reasoning_content` across assistant/tool turns.
- Compare `--reasoning-budget 0` vs `--reasoning-budget 256`.
- Keep temperature, max steps, model, tools, context size, and generation cap fixed.
- Start with deterministic/non-live-search diagnostic cases where possible.
- Do not run GAIA100 unless the bounded diagnostic produces promotion evidence.

This is an experiment contract, not a claim that 256 tokens is optimal.
