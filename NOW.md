# NOW — Small Agent QLoRA — 2026-11-16

Status: **METRIC / SOURCE-ISOLATION GATE**; model **Qwen3.5-9B** fixed.

- [Verified: Actions `38911211243`] GAIA validation/dev Evaluation100 Base **9/100**, v3B **18/100**, v3D **18/100**; one matched run, not a generalized gain claim.
- [Verified: `docs/gaia-v3d-results.md`] v3D finals **25/100** vs v3B **44/100**; 20 final→nonfinal versus 1 reverse; v3D over-searches.
- [Verified: Actions `38752732871`] Controlled synthetic200 Base 34.5%, v3B 52.5%, v3D 62%; not GAIA.
- [Verified: `docs/pro-research-gates.md`] MuSiQue: separate 8-task original-short-answer prompt test **0/8→0/8**, no success; 24 exposed holdout questions cannot be tuned again. v8 T4 stopped after 50/100 optimizer steps without recoverable physical checkpoint.
- [Verified: repo tree] GAIA scorer is project-local quasi-exact replica; no recorded official parity test, full ≥3-seed matched baseline, or comprehensive source-decontamination report. Failure labels are heuristic proxies.
- [Blocked] Prior artifacts for per-Level v3B/v3D rates not available in current checked-in summary. No new 3-seed physical runs or adapter promotions claimed.

**One selected next experiment:** M0 (CPU-only audit of official-scorer parity, exposure/source hashes and error taxonomy without opening new answer labels); then owner-confirmed sample gate before any model calls. See `IDEAS.md`.
