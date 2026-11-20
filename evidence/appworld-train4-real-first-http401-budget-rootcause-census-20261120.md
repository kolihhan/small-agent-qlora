# AppWorld TRAIN4 original Qwen9B: first observed failure, real HTTP status and budget censoring (2026-11-20)

**PHYSICAL FOUR-SHARD OBSERVATIONAL STUDY COMPLETE; NO COUNTERFACTUAL OR ACCURACY GAIN.**

## Frozen original source provenance

- [**Physical GitHub Actions #44660003397**](https://github.com/kolihhan/small-agent-qlora/actions/runs/44660003397), all four original CPU Qwen9B TRAIN shards, source lock and aggregate SUCCESS.
- Official safe JSON aggregate artifact **14594944841**, four per-shard artifacts 14595818069 (0), 14596316161 (1), 14596156087 (2), 14595681977 (3). Original Qwen3.5-9B GGUF SHA256 `8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54`, pinned original AppWorld source commit `42b5bcf3cd334fee33f0c37c02070a9f5807add5`, fixed eight-scene roster source hash `5e689bb15c5281dd3b5ab25ccacd6d3cac4c3cc9109948e56bf1d486558e0c`; original first four task hashes only.
- Original AppWorld tool scaffold: 20 model steps (last final-only), 384 max new tokens per step, 48 API calls, 2 tools (`appworld_docs`, `appworld_call`). Original model prompt and tool-visible HTTP responses **unchanged**. No training/adapter/GAIA.
- Per-step raw HTTP exception *bodies* never exported; only numeric statuses, categorical error codes, abstract docs/API call order, official task score and anonymous source hashes. Public upstream authentication source identifies 401 as missing/invalid/expired or unauthorized bearer token/endpoint; this diagnosis is not equivalent to proving the exact missing parameter for each original request.

## Measured four-scene actual source facts

| Fixed TRAIN scene index | First observed tool error | HTTP statuses | Model turns | Last step forced final? | Official task success | Official nested state |
|---|---|---|---:|---|---|---|
| 0 | HTTP_ERROR step 3 | 401 ×7 | 17 | No | Fail | 1/2 |
| 1 | HTTP_ERROR step 6 | 401 ×10, 422 ×2 | 20 | **Yes** | Fail | 1/7 |
| 2 | APP_DENIED step 1 | 401 ×3 (later steps) | 20 | **Yes** | Fail | 2/8 |
| 3 | HTTP_ERROR step 2 | 401 ×2 | 8 | No | Fail | 3/9 |
| **Total** | — | **401:22, 422:2** | — | **2/4** | **0/4** | **7/26** |

All **24 actual AppWorld API calls failed** in this cohort; no successful business API responses. `appworld_docs` often succeeded as an information lookup, which is not accomplishing the task. Other tool events include 1 APP_DENIED and 3 blocked exact duplicates. First observed error type is not a verified *causal* first blocker. Repeated 401 calls are dependent events within just 4 tasks, **not 24 independent observations**.

## Independent causal diagnosis, confounders, and next admissible test

1. **Wrong initial emphasis falsified:** HTTP401 is 22/24 (91.7%) of actual API failures. Ordinary 422 field hint is not the dominant observed failure mode. 422 occurred twice late in one case, not as its FIRST observed error. This does not prove the model selected the wrong API parameter or lacked a required token; 401 can also signal invalid/expired token, wrong permissions or optional bearer wrongly passed.
2. **Public source feedback is masked by the bridge:** upstream AppWorld authentication handler gives a generic but actionable missing/invalid/expired token diagnosis; `AppWorldCallTool` shows only `HTTP 401. Check the API documentation and parameters`. Model cannot see even that public auth context.
3. **Budget confound established, effect not:** two of four source rollouts reached model step 20 where `AgentRuntime` forces `tool_choice='none'` and precludes more operations. It is improper to claim 'would succeed at 40 steps' without the same-model budget A/B; one scene exited at 8 steps and one at 17, so step cap cannot explain all failure.
4. **384 token cap unknown in these AppWorld observations:** the server request specifies max_new_tokens=384 but the current source artifact does not capture per-turn `finish_reason`/usage. Do not call it exhausted from the configured limit. Historical K8 Text2SQL length stops are a DIFFERENT runtime, not causal AppWorld evidence.
5. **Same-task repeat previously diverged:** temp0 does not ensure an identical unforced trajectory, especially when AppWorld model runs are initialized separately. Exact-prefix/replay must verify both tool-visible history and private simulator state to be a causal A/B.
6. **Tool budget × docs overhead:** scenes 0,1,2 repeatedly queried `appworld_docs` and then issued unauthorized API requests. This could reflect missing auth planning, absence of credentials, wrong API decisions, aggressive system prompt, or lack of budget to recover. The first tool error is merely a symptom until a treatment restores task state changes and success.
7. **No GAIA/v7 conclusion:** these TRAIN4 AppWorld baseline results use a distinct two-tool scaffold; they do not establish either QLoRA efficacy or generalization to GAIA.

### Decision

- Do not start QLoRA or enlarge doc payload globally.
- A **specific public-doc-grounded 401 hint** (required/optional `access_token`, missing-vs-present boolean, no raw token) is justified for a *feasibility test*, not default production deployment. Only try on a first-error 401 with public token contract and a new model turn.
- Same exact original Qwen, same 20-step/384-token limit, same tool APIs, same task, exact-prefix replay and private DB-state checkpoint equality. Report paired official state success, rescue/regression denominator, computational cost, and rate of actual eligible auth errors. Refrain from 'root cause solved' if only HTTP count decreases.
- No retraining and no HOLDOUT/GAIA score claims.

**This evidence is NOT an A/B.** It is a source/behavior precondition audit for deciding if a narrowly targeted causal experiment merits CPU.
