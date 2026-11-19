# Small Agent v7 / 2Wiki16: evidence-state prompt A/B — completed negative (2026-11-19)

## Immutable evidence
- Original Actions run [#44477751745](https://github.com/kolihhan/small-agent-qlora/actions/runs/44477751745): completed success; artifact `v7-2wiki16-evidence-AB-FINAL` (14509055969).
- Source `a51699dcd2b470b36e162e5145985a6f9099d3254ac7906a0666e6519e6550d5`; 16 frozen identity/support-disjoint 2WikiMultihopQA validation tasks, public non-GAIA.
- Same original Qwen3.5-9B GGUF, **pre-existing v7 adapter**, 12-step unchanged AgentRuntime and same local search/read tools. No QLoRA training, new model, external teacher, GAIA or hidden benchmarks. Inference artifacts sealed before publisher answer scoring.

## Actual complete paired results (n=16)

| Metric | Original prompt | Evidence-state prompt | Delta |
|---|---:|---:|---:|
| **Raw exact** | 3/16 | 0/16 | -3 |
| **Gold-blind last-line normalized exact** | **13/16** | **10/16** | **-3** |
| Completed | 15 | 14 | -1 |
| Tool calls | 128 | 146 | +18 (**+14.06%**) |
| Duplicate blocks | 6 | 13 | +7 |
| Tool errors | 0 | 0 | 0 |
| Annotated support-document reads | 56 | 64 | +8 |

Paired normalized correctness: **1 rescue, 4 regressions, net -3**.
- Normalization is the **frozen gold-blind last-line/answer-prefix/markdown removal** in `scripts/run_hotpot_original_prefix_fork8.py`. Raw vs normalized scoring is fundamentally different; never combine.
- Predeclared screen: net at least +3, regressions at most 1, tool calls at most 1.5x. **FAIL**. `positive_screen_only=false`, `promotion_allowed=false`.
- More observed source-reading is not equivalent to more correct multihop reasoning. This prompt increased searching and worsened final answer correctness. Do **not** merge or promote the evidence-state prompt.
- Limitation: public 2Wiki task overlap with pretrained knowledge cannot be disproved. The run tests **v7 and 2Wiki**, not historical v3B GAIA82; it is not a statement about GAIA improvement or new weight learning.

## Next research implication
Stop treating additional read coverage or tool usage as a target. Investigate *which completed answers regress after reading enough facts* on fresh error traces and **separate final-answer selection/formatting from evidence acquisition** in factorial evaluation before any new QLoRA investment. Keep fixed model/protocol. Only propose changes based on actual scored paired task success and regressions.
