# Fixed original Qwen3.5-9B + v7: new public 2Wiki16 final caption screen — negative (2026-11-19)

**Original complete run:** [GitHub Actions #44486630457](https://github.com/kolihhan/small-agent-qlora/actions/runs/44486630457) `completed: success`; all 4 original physical Qwen/v7 worker shards completed, final aggregate completed, official artifact `v7-next16-zero-call-output-contract-FINAL` (ID `14511307701`). This is the second **new source-disjoint** 16-question experiment, not the earlier 16 used to diagnose the three literal-caption failures.

**Frozen source SHA:** `79b2413cec3c6ecf317e2dc9157c9f64f4d8f572003a82ebe6d0f079572fd47a`; previously exposed 40+48+8+16 identities and annotated support paths excluded by original ranked selection. Public 2WikiMultihopQA validation; pretraining exposure and other unrecorded overlaps remain unknown.

## Whole original 16, one real v7 agent inference per task

| Measure | Count |
|---|---:|
| Native original quasi-exact GAIA-style scorer on the **verbatim full answer** | 4/16 |
| Original frozen gold-blind last-line normalizer | **14/16** |
| Prospective additional anchored `The final answer is` / `The answer is` removal | **14/16** |
| **New cleaner rescues / regressions / net** | **0 / 0 / 0** |
| Actual completed / all tasks | 15/16 |
| Real original tool calls | 134 |
| Duplicate blocks | 4 |
| Annotated support-document paths successfully read | 62 / 64 possible |

**Preregistered gate:** >=3 net rescues and zero regressions vs frozen old normalizer, on exactly 16 questions. **FAIL**. No model retraining or changes to the runtime. Zero extra inference calls for the cleaner. Do **not** promote a parser rule based on older 16 exposed cases: their apparent +4 was a posthoc formatting observation, not independently replicated.

## Two actual failures remaining (original task source & publisher answer held out until scoring)

1. Hash-prefix `d9c801f4088b`: asks which film has the director born **later**, *The Golden Trail* or *Young Medardus*. Publisher answer *Young Medardus*. Original agent **read all four annotated supporting article paths**, but produced empty final with stop reason **`invalid_final`**, 16 tool calls. Missing answer is a terminal/tool-control failure, not evidence that the source article was inaccessible. Source coverage alone did not make it finalize.
2. Hash-prefix `b32bf86a0896`: asks which film has the **younger director**, *Dead Daughters* or *Terminator Salvation*. Publisher answer *Dead Daughters*. Agent's completed response explicitly stated **Pavel Ruminov born 1974**, **McG born 1968**, then wrongly concluded McG was younger and selected *Terminator Salvation*. It read 2 of 4 annotated support paths, completed with 6 tool calls. This is an **inverted numeric/relational comparison within the model's own answer**, not an answer-caption formatting error. The birth years it quoted imply 1974 is younger than 1968, assuming the compared people are the relevant directors.

## Next action now running

[New genuine identical-prefix step12 JSON final experiment #44504447550](https://github.com/kolihhan/small-agent-qlora/actions/runs/44504447550): 32 additional, source/support-disjoint public 2Wiki questions; SHA `de51bcdae148a4bee7867516c00d2841ef05a6fed96c94da9348b774549d98b9` frozen **before model inference**. The same original Qwen3.5-9B + v7 model and unchanged 12-step AgentRuntime performs a single complete tool trajectory. **Only if step12 is reached**, it compares the original final decode with a second same-model JSON-schema constrained final answer on the **identical messages**; no extra browsing or new agent tools. Publisher labels physically isolated until all original and structured outputs are sealed. Metric is complete paired correct answers, rescues, regressions, added model calls and finalization; not just valid JSON.

**No modification** to production runtime or adapter and **no GAIA82/genuine generalization claim**. Any positive public source screen must be independently confirmed, including GAIA exposure restrictions, before promotion.
