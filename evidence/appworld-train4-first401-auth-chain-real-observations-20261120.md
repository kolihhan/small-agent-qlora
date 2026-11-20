# Real original Qwen3.5-9B AppWorld TRAIN4: first-HTTP401 authentication action-chain and model error typing (2026-11-20)

**Verified COMPLETE observational study. No counterfactual treatment, no official task-success improvement, no adapter training.**

## Frozen real evidence / original constraints

- [Actions **#44675856600**](https://github.com/kolihhan/small-agent-qlora/actions/runs/44675856600): source lock, all 4 original Qwen task shards and aggregate **SUCCESS**. Authoritative aggregate artifact **14602611793**, individual source-safe JSON shards **14602552304** (#0), **14602850530** (#1), **14602209047** (#2), **14601763925** (#3).
- Original Qwen3.5-9B GGUF digest `8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54`, pinned original AppWorld commit `42b5bcf3cd334fee33f0c37c02070a9f5807add5`, same frozen four TRAIN source task IDs. 20 model steps, 384 max new tokens per model turn, 48 allowed API calls.
- Instrumentation observes only *original* model/tool trace **inside private task runner**. Export contains safe booleans, bounded step counts, numeric HTTP status, and finite typed model-error category. Never exports task prompt, app/API name, arguments/credential values, model text, raw error body or hidden evaluator content. Original model-visible tool feedback and task prompt unchanged.
- New typed error classifiers cover transport/timeout, HTTP 4xx/5xx, server JSON/choices/message malformation and model tool argument JSON/type. Legacy stop reasons retained; synthetic tests CI [#44675915440](https://github.com/kolihhan/small-agent-qlora/actions/runs/44675915440) SUCCESS. **No model_error event occurred in this new 4-task run**, so the precise subtype of *earlier different run's* model_error remains unknown.

## Actual official outcomes

**Official AppWorld task success 0/4; nested state tests 7/26.** Total real API errors 22: **401×20, 422×2**. 2/4 tasks reached forced final-only step20. No successful AppWorld business API calls. Compare [earlier same-four-task observation #44660003397](https://github.com/kolihhan/small-agent-qlora/actions/runs/44660003397), which had 401×22, 422×2. Difference is repeated model trajectory instability despite temp0, NOT an intervention gain or change in scoring.

### First actually observed 401 — state before the request

| Scene | First HTTP 401 step | Relevant API docs actually viewed, token requirement visible? | Same-app login API docs viewed? | Same-app login API called before 401? | Access token supplied? | Safe first-401 diagnostic | Steps remaining | Official success |
|---|---:|---|---|---|---|---|---:|---|
| 0 | 3 | **No** | No | **No** | **No** | Required `access_token` absent | 17 | Fail |
| 1 | 6 | **Yes** | No | **No** | **Yes, nonempty** | Required `access_token` supplied but rejected | 14 | Fail |
| 2 | 15 | **Yes** | No | **No** | **Yes, nonempty** | Required `access_token` supplied but rejected | 5 | Fail |
| 3 | None | N/A | N/A | N/A | N/A | Finished after 5 model turns without any API call | N/A | Fail |

Scene2 had an **earlier separate first observed tool failure APP_DENIED** at step1, before its HTTP401 at step15. Thus the first 401 is **not always the first tool failure or first causal obstruction**.

### What can and cannot be inferred

1. **Confirmed:** In 3/4 original tasks, no same-app `login` API call was observed **before first 401**. Two requests sent nonempty tokens without a preceding successful login call in this task, and one attempted an endpoint without any token after not reading that API's docs. This differentiates lack of relevant documentation discovery (scene0) from missing authentication prerequisite after docs were already seen (scenes1/2).
2. **NOT confirmed:** Those two supplied tokens were invented by Qwen. Token values are intentionally not retained or exported, so origin (user task text, earlier protected context, invalid/expired tokens, or fabricated) remains unknown. Nor is a public login always sufficient for authorization/permissions.
3. **NOT confirmed:** The `20-turn` budget caused failure. Scene0 still had 17 steps after first401 and scene1 14; scene2 only 5. Step20 forcing final in 2 cases is a confound, not causal proof.
4. **NOT confirmed:** All three 401s share one fixable policy error, or QLoRA will learn it and transfer to GAIA. AppWorld source tasks have already been examined; only four source tasks, repeat-run paths unstable. The before-401 booleans do not say whether login happened AFTER the first failure or whether task credentials were truly available.
5. **NOT confirmed:** The earlier `model_error` of strict public-auth feedback A/B [#44662846202](https://github.com/kolihhan/small-agent-qlora/actions/runs/44662846202) was JSON/HTTP/context-related. This run has 0 real `model_error`; typed classifier is tested but not yet triggered under the same paired treatment.
6. **Relevant negative evidence:** Public auth docs are not systematically hidden. Pinned source audit [#44667209288](https://github.com/kolihhan/small-agent-qlora/actions/runs/44667209288) showed **372/372** required-access_token fields visible *if* docs are requested. Real scene0 did NOT view the relevant API docs; scene1/2 did.

## Reusable causal decision gate, not a generic prompt change

Before another expensive original-Qwen repetition, first add a PRIVATE-only token provenance comparison: bool whether supplied token was present in user task text (length-capped equality check) and/or prior successful auth response, **without exporting any secret or its hash**, and verify public login prerequisites/credential availability. That discriminates (A) missed docs/login, (B) token unavailable from task context, (C) invalid/rejected supplied credential, (D) early final/step-budget limitation. Also check login attempted AFTER first 401, not only before.

Only if a repeatable source-verified mechanism has nonzero exposure should an exact-prefix **same saved private AppWorld DB state** A/B be run, with same Qwen9B/token/turn budget, predeclared official Task Success rescue/regression and costs. A source-grounded login-dependency action planner differs from already-failed plain public 401 explanatory hint (one valid pair, 0/1→0/1), but no causal success demonstrated yet. No production merge, QLoRA training, or GAIA transfer claim.
