# Small Agent QLoRA — action-value verification gate (2026-11-19)

Source: original Qwen3.5-9B natural HotpotQA first-read fork, source run 44466377952. This audit definition was recorded **before the final 8-task aggregate was available**. Do not use it to resplit or selectively suppress failed tasks.

## Distinguish observed correctness from action utility
- **Valid prefix:** original model actually called search/read, both branches replay identical byte-hashed delivered tool results, same model/tools/workspace and total 12-step budget.
- **Action-mediated result (strict):** continue-tools branch must issue an additional tool call AFTER the prefix, that call must succeed, and the model must receive genuinely new source evidence; the answer-now branch receives no postfork tool result.
- **Potential effect:** paired raw and independently locked gold-blind normalized exact-answer results. If the continue branch makes **zero additional calls**, differences are decoding/format variance, not proof that additional tools helped.
- **No confirmed training label** without new evidence successfully observed and an independently verified correctness transition. Even a branch score rescue may be an A/A fluctuation.
- **A/A baseline:** at least three independent model repeats on the same immutable prefix, reporting answer variance and tool-choice variance; temperature 0 is not guaranteed deterministic on all servers.
- **Transfer gate:** replicate on a different non-GAIA task family with intact reference evidence and native search/read/python tools; source-disjoint within HotpotQA does not imply family-independent GAIA transfer.

## Decision
- Existing declared pilot criterion (>=6 paired, normalized NET >=2, <=1 regressions) is a **score screen only**, not training authorization.
- Train QLoRA **only** from verified action advantages after source-family replication, with finalization and format controlled, not from raw hypothetical preference or any GAIA answer.
- If tool activation is low, stop the experiment rather than spending GPU on unverifiable stop/tool labels.

No QLoRA, GAIA tests, stronger teachers, or model swaps are authorized by this note.
