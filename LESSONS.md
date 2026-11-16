# Lessons — QLoRA agent

1. [Verified: `docs/v3d-results.md`] A gain on 200 controlled synthetic policy tasks (v3B 52.5%→v3D 62%) does not guarantee a gain on GAIA (18%→18%).
2. [Verified: `docs/gaia-v3d-results.md`] v3D GAIA finalization regressed 44%→25%; 20 paired final→nonfinal vs 1 reverse, with no decrease in tool errors. Treat termination as a mechanism hypothesis, not proof that a post-hoc stop rule improves correct answers.
3. [Verified: `docs/pro-research-gates.md`] MuSiQue long-form tool use fails despite access to relevant text. In one 8-task cohort only 2/8 visited every annotated supporting path. Reading a literal or formatting an answer differently alone is not a correctness gain.
4. [Verified: `docs/pro-research-gates.md`] Previous short-final prompt on last8 MuSiQue cohort remained 0/8 paired; do not rerun it on now-exposed 24.
5. [Verified: `docs/pro-research-gates.md`] Historical v8 T4 progress 50/100 optimizer steps without durable adapter is a resource failure, **but a separate newer NextSearch 8-step T4 adapter IS physically verified** (run `43732429711`, SHA `69deffc3ca0548778cf451332b76ee19073780d0490baba22193a36ed19c3508`). Its paired healthy 2Wiki VAL4 test was 0/4→0/4 with identical tool counts and support-path coverage (run `43744326618`). Do not conflate training success with benchmark success.
6. [Inference] Longer evidence-chain decision supervision is more plausible than another short synthetic stopping-only curriculum, but the causal mechanism and labels must be validated before GPU.
7. [Unresolved] Current script is a local approximation to official GAIA normalization, repeated-seed score variance and objective taxonomy-label agreement are unmeasured.
8. [Policy] No stronger model at inference. Only the pinned Qwen3.5-9B base plus controlled prompt/scaffold/tool/decoder/adapter changes; stronger teacher data require owner consent and disclosure.

9. [Verified: Actions `43836875211`] An actual official scorer revision and local implementation agreed on all 123 synthetic numeric/string/list/None cases (0 mismatches, no GAIA data). Scorer agreement is separate from model accuracy; it does not make a reused GAIA validation subset a held-out test.

Five-round decision audit: reopen any prior all-or-nothing verdict based on n smaller than the owner-approved gate as **insufficient evidence**; preserve negative evidence for the exact tested mechanism.
