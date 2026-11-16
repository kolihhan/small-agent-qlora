# QLoRA Small Agent metric contract — 2026-11-16

## Objective, model and scorer

- Target: better **GAIA 2023 validation DEV** exact correctness with fixed **Qwen3.5-9B** base and independently verified tool-agent behavior. The historical Qwen3.5-4B early TinyAgent experiment is not the current base. No stronger inference-time model. Stronger teacher data require prior explicit consent and disclosure.
- [Verified: Actions `43836875211`] The actual official GAIA leaderboard scorer at revision `1349a17` (SHA256 `0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2`) was executed against the repository implementation on **123 synthetic edge cases**, with **0 mismatches**, **no GAIA data loaded**. This is a *limited parity check*, not proof for all inputs or a new GAIA result.
- Scoring code: `src/gaia_small_agent/benchmark/scoring.py::gaia_score` mirrors official GAIA quasi-exact-match branch order: numeric answer comparison, comma/semicolon-delimited aligned list comparison, or normalized lowercase, no-whitespace, no-punctuation string comparison. This is a **locally implemented approximation**; an upstream scorer parity test is still required for an official-equivalence claim. No LLM judge.
- Official score numerator: correct final answers; denominator: frozen distinct task IDs. Never use completion as a replacement. Log Level 1/2/3 denominators, accuracies, per-run and pooled statistics, answer formatting failures, and tool/cost metrics.
- Existing frozen Evaluation100 (within GAIA validation) uses Level quotas **32/52/16**, revision `682dd723ee1e1697e00360edccf2366dc8418dd9` and selector in `src/gaia_small_agent/benchmark/gaia100.py`. Existing "evaluation" and "shadow" names in code are **all GAIA validation/dev**; they are NOT an independent out-of-sample test set and no repeated use can make them a holdout.

## What has actually been measured

| Cohort | Base | v3B | v3D | Provenance |
| --- | --- | --- | --- | --- |
| GAIA validation frozen 100, one matched deterministic run | 9/100, 45/100 final | 18/100, 44/100 final | 18/100, 25/100 final | Actions `38911211243`, `docs/gaia-v3d-results.md` |
| Controlled deterministic 200, synthetic | 69/200 | 105/200 | 124/200 | Actions `38752732871`, `docs/v3d-results.md` |

- [Verified: `docs/gaia-v3d-results.md`] v3B vs v3D: both right 13, v3B-only 5, v3D-only 5, both wrong 77; **no new GAIA correct answers in aggregate**. Final→nonfinal 20 vs nonfinal→final 1, exact paired McNemar p≈2.1e-5 for **finalization only**, NOT correctness. Tool calls v3B 784→v3D 876; duplicate blocks 66→127.
- Level-level correctness for that 3-arm historical run is **not present in the checked-in aggregate note**. Do not invent; fetch task-level artifact before asserting by-level percentages.
- [Verified: `results/final-gaia-v2-summary.json`] An **older and different** matched protocol recorded Base 12/100 and QLoRA 8/100. Do not pool/mix with v3B/v3D above.
- [Verified: `docs/pro-research-gates.md`] Historical v8 training had no exported checkpoint after 50/100 steps. **Separate NextSearch 8-step T4 adapter** was physically completed (Actions `43732429711`, SHA `69deffc3ca0548778cf451332b76ee19073780d0490baba22193a36ed19c3508`), yet source-disjoint healthy 2Wiki VAL4 strict exact stayed **0/4→0/4** (Actions `43744326618`). No GAIA score exists for this new adapter; no generalization claim.

## Statistical and repeatability gates

- [Not yet met] Frozen identical runtime/scaffold/tools, budget (steps=12, max_new_tokens=512, timeout/context fixed), immutable model/adapter digests, temperature, data revision, and task IDs for **≥3 independent seeds/config repetitions**. Run no higher-stakes evaluation until precise manifest/cost approval; store complete trajectories per repetition.
- Report by Level: each run exact successes / n, mean and sample SD across runs, exact pair discordance, paired bootstrap (task-ID clusters, 10,000 resamples) 95% CI and McNemar exact p for matched configurations. For repeated seeds, also bootstrap paired **task ID** across runs and report run-to-run variability separately; do not treat 3 repeats of the same task as n=3 independent tasks.
- **Proposed gate (owner confirmation pending):** at least **200 distinct matched tasks** for a decisive large-cohort directional comparison; at least **30 discordant pairs** and bootstrap 95% CI excluding 0; independent task-family replication before production promotion. For GAIA only use the available validation data as exploratory DEV. If n/pairs are below the gate: **INSUFFICIENT EVIDENCE**, not universal rejection.
- **A/A variance:** with exact same serialized model/input/runtime, run at least 3 separate seeds; capture exact final answer, individual score and frequency of format/stop changes. Fixed temperature=0 does NOT guarantee zero variance across server environments. Score variance currently **not quantified**.
- Non-LLM grading, so human agreement is N/A for exact-answer scoring; qualitative failure-label inter-annotator agreement is NOT measured and must be separately estimated.

## Failure taxonomy and diagnostics

Primary mutually exclusive label with additional overlapping evidence flags: malformed tool call; wrong tool/arguments; search/read/transport failure; attachment/capability gap; reasoning/evidence selection error; incorrect final-answer format; step limit/duplicate looping; long-context overflow. Existing `benchmark/runner.py::classify_failure_label` is a **rule-based proxy**, not validated causal assignment. Report primary counts and % of failed cases, unresolved cases, and double-review agreement before treating taxonomy as fact.

Known actionable *hypothesis* from historical trace: long searches with missed support paths and missing finalization. MuSiQue 8-task audit from PRO notes found 2/8 complete support-path visits in a diagnostic cohort; reference contained in passage is not proof the model used it. One prior generic short-final intervention on different 8 tasks produced 0/8→0/8 and must not be repeated on exposed questions.

## Isolation / contamination / training

- All GAIA public validation/diagnostic/shadow/evaluation partitions are **development**, ever-seen task IDs must have an exposure counter. Preserve distinct study cohorts; do not adjust selection until a score looks good.
- QLoRA training source: verified non-GAIA tasks with distinct source families; exclude all GAIA question text, answers, attachments, near duplicates, and BIRD DEV questions/SQL. Audit exact hashes + normalized approximate similarity against complete reference corpora and inspect flagged pairs manually; **without full reference corpora, decontamination is UNVERIFIED**.
- Four factorial arms, fixed frozen tasks: 00 base scaffold + base model; 10 prompt-only; 01 adapter-only; 11 prompt+adapter. Log base SHA, adapter SHA, train source SHA, prompt SHA, decoding params, tool schema, tool budget, seed, training cost. Require tool-JSON validity, general task and format regression suite after training.
- Dataset split frozen in manifest, holdout never used to tune; GAIA validation cannot be described as untouched independent final holdout. No inference-time stronger model, no teacher without explicit prior permission.

## NextSearch TRAIN48 exact/near contamination evidence (2026-11-16)

[Verified: GitHub Actions `43844291341`; `evidence/nextsearch48-cross-benchmark-screen-20261116.json`]: pinned training source `5420cfdb6111806a23b4bf6cb9cf04c77c75b5f0cf4af6c0b63727f312f40432` contains **48 source action windows / 28 parent questions**. Against SHA-pinned original GAIA validation165 + test301 question metadata at revision `682dd723ee1e1697e00360edccf2366dc8418dd9` and all BIRD DEV1534 questions: **0 exact/strong-lexical-near parent-question matches**. Second pass scanned **146 unique complete training user/assistant/tool/system message/answer texts**, finding **0 embedded exact full-reference GAIA questions or BIRD DEV questions/SQL**. Both are lexical screens, not semantic paraphrase, attachment, or independent answer-success validation. Training labels remain sourced from publisher rather than re-executed successful agent episodes. This is **NO GAIA gain / NO ADAPTER PROMOTION**.

## Exposure ledger

| Source | Known exposure | Status |
| --- | --- | --- |
| GAIA 2023 validation Evaluation100 | historical Base/v3B/v3D matched run + prior experiments | Seen DEV, true cumulative count not reconstructed |
| GAIA Diagnostic25 and Shadow25 | prior development workflows/forensic use | Seen DEV; count requires artifact-ledger reconstruction |
| MuSiQue source train-derived cohorts | ≥3 separate 8-task experiments plus audits; known 24 prior holdout questions exposed | No further tuning on these tasks |
| New independent task-family test | Not frozen | BLOCKED until auditable source and labels exist |

For every new run append task IDs hash, unique exposure number per cohort, artifacts, seeds, model/adapter/config hashes, and whether outputs/references were inspected.

## Cost boundary

Before experiments estimate `tasks × seeds × max_steps × max_new_tokens` upper-token bound, expected tool calls/latency, GPU-hours, and available budget. No automatic Colab retries or paid GPU runs. No GPU allocated in this documentation-only round.


## Native40 and exact-answer rendering confound, verified 2026-11-16

- [Physically trained native40 adapter](https://github.com/kolihhan/small-agent-qlora/actions/runs/43805908527): T4 8/8 optimizer steps, original fixed Qwen3.5-9B, SHA256 `678a9515d56757e7740c9fbe774ae96d37dad79155b1860b4ac016d1fefa69b0`. Training completed; no task accuracy implied.
- [Physically matched closed-loop 2Wiki VAL8](https://github.com/kolihhan/small-agent-qlora/actions/runs/43892091237): 8/8 original frozen public source cases, Base **1/8** strict exact, native40 adapter **2/8**, paired 1 rescue/0 regressions. Both arms 58 successful local search/read calls, 0 errors, six tasks with all 4 annotated support docs observed. Public source already exposed in related prior evaluations; not hidden and **not GAIA**. Native40 has no validated GAIA score.
- [CPU-only immutable after-the-fact rendering audit](https://github.com/kolihhan/small-agent-qlora/actions/runs/43901267208): original raw replies and reference answers independently loaded by the scorer; extraction function reads **only the last nonempty assistant output line** and removes an explicit Answer prefix and paired Markdown, never gold/question/support. Same 8 previously scored replies: **Base 1/8 raw → 7/8 post-hoc**, **Native40 2/8 raw → 8/8 post-hoc**; **6 format-only rescues per arm**, 0 rendering regressions. Source/original answer hashes, original scores and both paired arms validated, 10 CPU tests passed. Last-line results are diagnostics and must NEVER replace stored original GAIA/2Wiki scores. Non-standard output extraction may corrupt numeric/list or explanation-ending questions, and is not shown generalized here.
- **Research decision:** do not extend LoRA steps or call +1/8 a reasoning breakthrough. First freeze a genuinely source-independent task family and evaluate the same unmodified answer contract under unchanged base/adapter inference, score raw and extracted side-by-side; preserve per-case evidence acquisition, finalization, reasoning-vs-format errors. A broader independent source is required before any GAIA claim.


## Completed physical v7 generic-format GAIA100 matched replication (2026-11-16)

[Verified successful 20/20 inference shards and strict aggregate](https://github.com/kolihhan/small-agent-qlora/actions/runs/43865621132): pinned Qwen3.5-9B GGUF and same v7 adapter in both arms. Frozen previously exposed GAIA validation Evaluation100; exactly **100/100 paired** results on original vs generic format reminder, not hidden and not new adapter weight training.

| Metric | Original v7 | Same v7 + format reminder |
| --- | ---: | ---: |
| Strict task success | **26/100** | **26/100** |
| Level1 (32) | 11 | 9 |
| Level2 (52) | 13 | 15 |
| Level3 (16) | 2 | 2 |
| Finalized | 39 | 44 |
| Model timeouts | 2 | 10 |
| Invalid final | 54 | 40 |
| Tool calls | 790 | 737 |
| Tool errors | 51 | 58 |
| Duplicate blocks | 91 | 60 |
| Mean case latency (s) | 431.07 | 424.54 |

Matched paired success discordance: **8 original-only, 8 reminder-only**; no net success gain. Completion improved by 5 but correctness did not, tool errors and model timeouts increased. **Decision: NO PROMOTION.** Better proxy metrics cannot substitute for GAIA correct answers. Do not compare 26/100 v7 against historical v3B18/100 as causal improvement because runtime, model adapters and protocol differ between experiments. The earlier 2Wiki VAL8 post-hoc final-line extraction effect is a different, narrow public-task diagnosis, not a demonstrated GAIA fix.

## GAIA100 original paired stop/finalization taxonomy (verified 2026-11-16)

[CPU replay of 20 original physical shards, complete 100/100](https://github.com/kolihhan/small-agent-qlora/actions/runs/43906372492): original v7 26 correct, format reminder v7 26 correct; shared 18 correct; 8 original-only; 8 reminder-only; **66 both wrong**. On **those 66 jointly missed tasks**, original stop reasons were final-but-incorrect **9**, invalid_final **51**, model_error **5**, timeout **1**. Reminder: final-but-incorrect **14**, invalid_final **39**, model_error **5**, timeout **8**. The broader full100 original 39 finals/13 wrong finals/61 nonfinal; reminder 44 finals/18 wrong finals/56 nonfinal. A +5 final-count proxy was exactly a +5 wrong-final-count, with no net correct-answer gain. **Conclusion:** terminal-output reliability is an important failure bucket but generic format prompting did not improve correctness. Need trace-level tool/evidence/finalization annotation on source-independent data to distinguish insufficient facts from incorrect reasoning, output serialization and tool-budget failures. No new weights, hidden holdout score or causal explanation claimed; stop reasons are runtime categories only.


## Causal terminal-only JSON Schema GAIA8: protocol improves, accuracy gate fails (2026-11-17)

[Successful real pinned model / fixed-prefix run](https://github.com/kolihhan/small-agent-qlora/actions/runs/43918680792): **8/8 exact frozen previously seen GAIA DEV cases, 4/4 shards, aggregate SUCCESS.** Original pinned Qwen3.5-9B plus identical v7 adapter; same original tool transcript, tool-choice setting and system prompt. Experimental arm replays **only captured step12 final messages** with llama.cpp schema-constrained `{"answer":string}` decoding, then original unchanged GAIA scorer. Nonterminal cases are **not treated**. No new training, hidden test, or model swap.

| Matched observed result | Original | JSON Schema terminal-only |
| --- | ---: | ---: |
| Strict correct on full exposed 8 | **5/8** | **6/8** |
| Actually reached forced step 12 | 3 | same 3 |
| Finalized among those 3 | **0/3** | **3/3** |
| Correct among those 3 | **0/3** | **1/3** |
| Invalid final among all 8 | 3 | 0 |
| Previously correct early-stopping cases (untreated) | 4 | same 4 |
| Extra model timeouts | 0 | 0 |

**Scientifically relevant distinction:** format/protocol compliance **3/3** after forced decoding, but task success only **1/3**. Predeclared promotion-to-replication screen required >=2 true wrong→right rescues and no regressions; **FAILED**. Furthermore 4 prior-correct controls completed before step12, so absence of regression is not an actual negative-control test on previously correct terminal answers. One prior invalid case also stopped early on rerun and was untreated; the test is conditional on actual forced-terminal opportunity. Source selection was dependent on previously exposed outcomes, not an unbiased test. **Decision: no production/default change; PR #64 closed unmerged.** Earlier PR #63 reran entire agent arms separately and suffered tool-history differences, so it cannot isolate terminal decoding causally and was closed. Do not compare this +1/8 to historical GAIA100 v7 26/100 or 2Wiki LastLine 7→8/8; these differ in population/protocol.

## Fixed-prefix JSON terminal correct-answer regression control — not exercised (2026-11-17)

[Physical controls on two predeclared previously correct forced-terminal Qwen3.5-9B/v7 GAIA Evaluation100 tasks](https://github.com/kolihhan/small-agent-qlora/actions/runs/43925609335): source #43865621132 had **two strictly-correct original tasks with 11 tool calls** (step12 final). Their task hashes start `3f9b8f9b1b` and `e424150204`; selection was fixed before new inference, both Level2, 2023 GAIA dataset revision `682dd723ee1e1697e00360edccf2366dc8418dd9`. A dedicated physical pinned source+model run completed SUCCESS, original 2/2 vs constrained 2/2, but **forced terminal reached=0/2** because both original episodes finished early on this new run. The JSON arms were explicitly no-intervention, not decoder replays. Thus **no tested original-correct terminal cases, no evidence of terminal safety, no GAIA100 uplift**. Failure to reenter terminal boundary despite previous 11 tool calls demonstrates tool/answer trajectory variability even with same model/revision/temperature. Future test must capture an actually correct forced-terminal transcript as it occurs and replay it immediately, and control the sample size; no approval from untouched early finals.
