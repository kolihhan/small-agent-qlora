# NOW — Small Agent QLoRA — 2026-11-16

Status: **METRIC / SOURCE-ISOLATION GATE**; model **Qwen3.5-9B** fixed.

- [Verified: Actions `38911211243`] GAIA validation/dev Evaluation100 Base **9/100**, v3B **18/100**, v3D **18/100**; one matched run, not a generalized gain claim.
- [Verified: `docs/gaia-v3d-results.md`] v3D finals **25/100** vs v3B **44/100**; 20 final→nonfinal versus 1 reverse; v3D over-searches.
- [Verified: Actions `38752732871`] Controlled synthetic200 Base 34.5%, v3B 52.5%, v3D 62%; not GAIA.
- [Verified: `docs/pro-research-gates.md`] MuSiQue: separate 8-task original-short-answer prompt test **0/8→0/8**, no success; 24 exposed holdout questions cannot be tuned again. Historical **v8** T4 stopped after 50/100 optimizer steps without recoverable checkpoint.
- [Verified: Actions `43732429711`] **Different, newer NextSearch 8-step adapter PHYSICALLY EXISTS**: 48 source-backed actions / 28 parents, complete 4096 tokens; 8 T4 optimizer steps; adapter SHA256 `69deffc3ca0548778cf451332b76ee19073780d0490baba22193a36ed19c3508`. Do NOT conflate with failed v8 run.
- [Verified: Actions `43744326618`] Healthy local 2Wiki VAL4 comparison of original Qwen9B vs this adapter: **0/4→0/4** exact, 26 successful real tool actions per arm, 0 tool errors, same support coverage **2/4→2/4**. No external transfer demonstrated; n=4 insufficient for universal rejection.
- [Verified: `docs/pro-research-gates.md`] Three of four 2Wiki explanatory finals contained the publisher answer as a substring, but exact final-score was 0/4. Post-hoc answer extraction is NOT a new benchmark improvement.
- [Verified: repo tree] GAIA scorer is project-local quasi-exact replica; no recorded official parity test, full ≥3-seed matched baseline, or comprehensive source-decontamination report. Failure labels are heuristic proxies.
- [Blocked] Prior artifacts for per-Level v3B/v3D rates not available in current checked-in summary. No new 3-seed physical runs or adapter promotions claimed.

- [Verified: Actions `43836875211`; `evidence/gaia-official-scorer-parity-synthetic.json`] Local GAIA scorer matched the pinned official leaderboard scorer at revision `1349a17` for **123/123 synthetic cases**, **0 mismatches**, and **0 GAIA question/answer exposures**. This checks scorer implementation only, not GAIA model accuracy or end-to-end scorer parity on every possible input.

**One selected next experiment:** M0 (CPU-only audit of official-scorer parity, exposure/source hashes and error taxonomy without opening new answer labels); then owner-confirmed sample gate before any model calls. See `IDEAS.md`.
