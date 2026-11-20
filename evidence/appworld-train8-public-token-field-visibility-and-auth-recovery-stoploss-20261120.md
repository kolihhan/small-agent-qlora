# AppWorld TRAIN8: public token requirements are already visible (2026-11-20)

**Completed source-only audit; zero Qwen calls, no new official task score.**

- [Pinned real AppWorld Actions #44667209288](https://github.com/kolihhan/small-agent-qlora/actions/runs/44667209288), SUCCESS, artifact 14598206771.
- Eight scene-distinct TRAIN instances have a single observed documentation **count** profile, not eight independent public API catalogs.
- One representative scene: **10 public apps, 9 apps with public login, 469 public APIs**. **372 APIs require access_token**, and **all 372 access_token parameter markers remain in the actual model-visible document after the 2,000-char observation cap**. All 372 belong to an app with a public login endpoint. Another **4 optional-token APIs** have token field visible 4/4. Of 372 required-token docs, **11** exceed 2,000 chars somewhere, but access_token remains visible.
- The census calls the original AppWorldDocsTool then ToolResult.bounded_observation(2000) exactly as the agent would; it counts exact access_token parameter markers only. It exports no protected task instructions, API names, parameter values, credentials or raw failures. Public API visibility does **not** prove that Qwen actually requested that doc.

## Causal evidence boundary

- [Real original Qwen TRAIN4 #44660003397](https://github.com/kolihhan/small-agent-qlora/actions/runs/44660003397) scored official 0/4, HTTP 401×22 + 422×2.
- [Strict public-auth feedback A/B #44662846202](https://github.com/kolihhan/small-agent-qlora/actions/runs/44662846202) has **one** successfully matched model-visible prefix and private saved DB state, original official 0/1, treated 0/1. Treatment ended model_error (subtype not logged). Another scene had no eligible first 401.
- **Falsified narrow premise:** the generic auth parameter itself was hidden due to API doc omission/truncation. It is already visible *if the model asks for that API doc*. Mere 401 status can mean missing/invalid/expired token, authorization, or wrong workflow.
- No experiment has shown official AppWorld task success uplift, no inference that auth hints repair decisions, and no training label yet.

## Next controlled observation (not yet measured)

Before spending another Qwen task run, instrument privacy-safe boolean counters at the original first 401: relevant API docs actually viewed; access_token marker in delivered observation; login docs viewed; login attempted/succeeded; token supplied present/not present; typed model_error category rather than raw server text. Never export private API, user task, credentials, raw response, or hidden grader.

**Decision: stop repeating identical 401 hint; do not merge it as a fix or train QLoRA.**
