# AppWorld TRAIN4: two complete Qwen3.5-9B original-replay intervention gates + adversarial review (2026-11-20)

**VERIFIED NEGATIVE COVERAGE / NOT A NEGATIVE TREATMENT EFFECT.**
No adapter, no GAIA run, no policy merge. Public evidence contains only safe task hashes, error classes and official state scores.

## Frozen source and artifact identity

- Original AppWorld upstream SHA `42b5bcf3cd334fee33f0c37c02070a9f5807add5`.
- Original Qwen3.5-9B GGUF SHA256 `8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54`.
- Scene-disjoint fixed TRAIN roster SHA256 `5e689bb15c5281dd3b5ab25ccacd6d3cac4c3cc9109948e56bf1d486558e0c`; same four task hashes in both studies.
- Runtime constraints: up to 20 model turns, up to 48 AppWorld API calls, 384 generated tokens/call, llama-server temperature=0, CPU GGUF. Tool bridge exposes `appworld_docs` and `appworld_call`; it is NOT the benchmark's general Python-code interaction harness.
- Study A [HTTP422 documented-field-only original-vs-exact-prefix fork run 44565392162](https://github.com/kolihhan/small-agent-qlora/actions/runs/44565392162), four original shard artifacts `14554025209`, `14554105413`, `14551114357`, `14552555564`; official aggregate artifact **`14553260868`**.
- Study B [APP_DENIED public-task-app-list original-vs-exact-prefix fork run 44568286156](https://github.com/kolihhan/small-agent-qlora/actions/runs/44568286156), four original shard artifacts `14554457439`, `14552824823`, `14554056876`, `14552719080`; official aggregate artifact **`14554238740`**. Both Jobs + source integrity + strict aggregate SUCCESS.

## Physical official outcome per same task

| Shard | Study A first tool error | Study A original score | Study B first tool error | Study B original score |
|---|---|---|---|---|
| 0 | HTTP_ERROR at model step 3 | false, nested **1/2** | HTTP_ERROR at step 4 | false, nested **0/2** |
| 1 | HTTP_ERROR at model step 6 | false, nested **1/7** | HTTP_ERROR at step 3 | false, nested **1/7** |
| 2 | APP_DENIED appworld_docs at model step 1 | false, nested **2/8** | HTTP_ERROR appworld_call at model step 7 | false, nested **2/8** |
| 3 | no tool error; early final | false, nested **3/9** | HTTP_ERROR appworld_call at step 2 | false, nested **3/9** |

- Study A original official task success **0/4**, eligible documented `HTTP 422` field error **0/4**, exact-prefix paired denominator **0**, rescue/regression **not observed**. Underlying original AppWorld API requests **22**, recorded HTTP errors **22**. This says nothing about hypothetical fixes to other HTTP statuses or whether a protected exception payload contained a parsable public field.
- Study B original official task success **0/4**, FIRST `appworld_docs.APP_DENIED` eligible episodes **0/4**, exact-prefix paired denominator **0**, rescue/regression **not observed**. Original AppWorld API requests **29**, recorded HTTP errors **25**. No treated Qwen calls actually happened.
- **Same task, SHA, model temperature=0, and code runtime settings still did not reproduce the exact original first-error path**: all four cases changed the step/type/absence of first tool error. In particular shard2 changed first APP_DENIED at step1 to first HTTP_ERROR at step7. Original model-call lengths changed (shard2 20→9, shard3 5→8). This is physical within-task replay instability, not a measured treatment improvement.
- Sum of reported raw original Qwen student wall times in artifacts: Study A ~**2040.88 seconds**; Study B ~**2465.76 seconds**. These are summed task model-runtime wall seconds across parallel workers, not directly billing seconds; installation/queue overhead excluded. **4506.64 sec ~75.11 worker-minutes** of original Qwen episode time, **zero eligible matched Qwen treatment continuations**.

## Independent reasoning audit: erroneous assumptions, gaps and confounders

1. **An observed first tool failure is not necessarily the first *causal* blocking point.** The model may already have misunderstood the user's objective, chosen the wrong plan or skipped task operations before its first HTTP error. To call a failure the blocking mechanism, a counterfactual must rescue at least an actionable downstream step and, separately, official task success.
2. **Neither hypothesized feedback cause passed the activation gate.** Study A required an HTTP 422 exception that parses to a public schema field; Study B required the very first error to be a docs APP_DENIED not preceded by an already visible task-app list. Both yielded denominator 0. This does **not** falsify their effect if present; it falsifies the premise that these four source scenes supplied evaluable evidence for the interventions.
3. **HTTP error status is unknown in the artifact.** The first run had 22 recorded HTTP failures but the experiment only reports no *eligible parsed 422*, not that none were HTTP 422. The safe extractor uses a strict exception format; missing or differently formatted public validation payload is a competing explanation. Instrument privacy-safe status histogram and parser-fail reason before making a new intervention claim; no raw protected response export.
4. **Temp0 does not guarantee deterministic agent rollouts.** Code sends `temperature:0` but no explicit server generation seed. API state/timing, sampling ties, server implementation or observations could affect the path. Only task ID hash and model digest were frozen, not a cryptographic snapshot of entire underlying simulator state. The fork implementation asserts model-visible prefix equality, not hidden database state identity.
5. **The scaffold itself is a confounder.** 20 turns, 384 tokens/turn, 48 API calls, docs and direct API only, the supervisor completion instruction, and sanitization of all HTTP exception bodies can limit AppWorld task success. A result of 0/4 (or historical DEV57 0/57) does not isolate a learned policy deficiency, let alone a QLoRA-fixable one.
6. **Operational metric is too weak.** `first_error_followed_by_correctly_scoped_app_call` checks whether a *later app name is in the allowlist*, not whether the original error's cause was resolved or an API state transition succeeded. It cannot be called an error rescue. Even a successful later API call would only be an intermediate proxy, not official task success.
7. **Cannot compare model variants.** No matched bare Qwen, v7 and new adapter AppWorld/GAIA official evaluation exists in this branch. Old v7 GAIA results and these TRAIN4 base results differ in scaffold and source; no causal v7 improvement claim.
8. **Adaptive selection and small denominators.** Four scene-distinct tasks do not establish a robust failure family; the new hypothesis was chosen after seeing an APP_DENIED in one prior run, which failed to recur. Need independent task-family coverage and paired event triggers, not ever-more targeted rematches.
9. **No useful training supervision exists yet.** No several independent fully informed wrong decisions and no stable validated alternative action have been observed. Small-agent QLoRA is **NOT APPROVED** under the user-given evidence gate. No GAIA new final-answer score.

## Disposition

- Mark both feedback-only mechanisms **NO TESTABLE EXPOSURE ON THIS ROUND**. Do not claim 0 paired rescues as a negative efficacy result; do not merge the treatment into default.
- Stop expensive repeat TRAIN4 Qwen9B runs until a **CPU-only** source/tool-contract audit verifies the true first HTTP status categories and current bridge's full public documentation affordances, plus safe summarized first model-visible error templates. Neither source artifacts nor this PR contain the raw protected traces to retrospectively infer status.
- Next new run only when an independently frozen first-obstruction category has measurable source-only exposure, exact-prefix model-visible input and **same simulator state** have sufficient checks, and success proxy and official Task Goal Completion are separately preregistered.
- Preserve the AppWorld reference, model identity and 60%/40% priority; withhold QLoRA until evidence gates met.
