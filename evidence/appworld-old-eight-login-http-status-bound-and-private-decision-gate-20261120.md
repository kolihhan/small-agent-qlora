# AppWorld auth diagnosis stop-loss: exact historical information boundary + future intervention decision (2026-11-20)

**This branch contains no new Qwen inference and no model-facing policy change.**

## Completed historical-only evidence

[Actions **#44711677586**](https://github.com/kolihhan/small-agent-qlora/actions/runs/44711677586), original safe source artifact from [#44679395972](https://github.com/kolihhan/small-agent-qlora/actions/runs/44679395972), ALL SUCCESS; offline source-only evidence artifact ID **14616479667**.

The original scene #1 retained exactly 12 API HTTP errors, one at first401 step3 and 11 after that. The 11 later statuses were 401×10 + 422×1. Its original private runner had counted exactly **eight post-first401 same-app login API tool calls**, all failed at requester, no duplicate-call/budget block. But uploaded anonymized per-error event rows have only `api_call_index`, `model_step`, numeric HTTP status and generic `appworld_call`, **not per-event API identity, call ID or parameter list**.

**Sharp feasible information bounds:**

| Login-attributed HTTP status | Min | Max | Status |
|---|---:|---:|---|
| 401 | **7** | **8** | at least 7 login calls must be 401 |
| 422 | **0** | **1** | 422 may or may not be a login request |
| All login HTTP errors | 8 | 8 | previously verified original login count |

Proof: eight login errors occupy eight of eleven post401 API errors; only one is 422, thus at least seven are 401. Nothing in uploaded source identifies *which* steps correspond to login. **Do not assert that the post401 step7 HTTP422 was a login**, nor infer missing login fields or invalid/expired credentials from generic HTTP status alone. Trying to re-execute the original fixed Qwen creates a new stochastic/state-dependent trace, not the missing old one. No raw private trace was retained in the artifact.

## Future PRIVATE-RUNNER-only per-call evidence, ready in code (NOT yet run on real Qwen)

`src/gaia_small_agent/tools/appworld_private_login_event_audit.py` is source-only instrumentation taking the original unexported `TraceEvent`, local public API docs and requester numeric HTTP events. It creates no new tool calls and changes no prompt. It binds a login tool call/result by local private call ID, then binds numeric HTTP status **only when exactly one API HTTP_ERROR occurred on that model step and one matching tool-result exists**; all ambiguous same-step matches output `AMBIGUOUS_OR_MISSING_CALL_HTTP_ID`. It checks against the *public required parameter names/types inside the runner*, then exports only number of missing/empty required fields, documented type mismatches, source-visible value occurrences as counts, HTTP status and step. **No parameter names, values, app/API names, task text, token or token hash exported.**

Synthetic [CI #44714059716](https://github.com/kolihhan/small-agent-qlora/actions/runs/44714059716) SUCCESS: 401/422 exact unique-step matching, ambiguous multiple same-step fail-closed, required public parameters, redacted sensitive values, and source-only historic sharp bounds. This success **does not reclassify historical eight login errors** because the original private full trace is missing.

## Intervention gate: exact triage conditions (no training yet)

| Private-first observation | Next decision | Why |
|---|---|---|
| Exact call-ID/status or public argument contract **not recoverable** | **ABSTAIN** | Cannot identify failed prerequisite |
| HTTP422 + documented required field missing/wrong documented type | **CONTRACT_FIX_HYPOTHESIS**, still not Task Success | Parameter construction may be the blocker; independently verify complete legal login contract |
| HTTP401 + all public required fields provided, with credentials of known legitimate source but rejected | **ABSTAIN/VERIFY_CREDENTIALS_OR_PERMISSION** | Login ordering alone is not sufficient |
| Protected API called with no valid token, login not attempted, **legitimate login prerequisites verified available**, relevant docs accessible, enough budget | **AUTH_DEPENDENCY_PLANNER_A_B_ELIGIBLE** | Tests missed dependency/sequence, *not* generic 401 explanatory hint |
| Login attempted and succeeded but later request still 401 | **TOKEN_SCOPE_OR_PROPAGATION_HYPOTHESIS** | Must verify same app, scope and result-extraction before model training |
| All attempts hit model-final budget | **BUDGET_CONFOUNDED** | Budget-only paired control necessary, not a login prompt attribution |

**Causality:** A/B on frozen same original Qwen (same 20 turns, 384 token max, 48 API calls), identical tool-visible prefix and persisted AppWorld DB checkpoint on original/fork. Primary endpoint official task success plus state tests, rescues/regressions and real cost; prompt-only HTTP count is not an endpoint. TRAIN scenes used repeatedly are pilot only; new source-disjoint test needed before claims of generalization. No QLoRA, production merge or GAIA improvement claim.

**Decision today:** historical 8-login one-to-one status/argument attribution is IMPOSSIBLE from retained artifacts; safe next-run instrumentation is implemented and unit-tested, **but no approval for another costly original Qwen rollout until a truly discriminating exposure is predeclared**.
