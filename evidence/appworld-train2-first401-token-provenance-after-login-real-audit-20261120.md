# Original fixed Qwen AppWorld TRAIN2: private first-401 token-origin and post-401 login recovery (2026-11-20)

**Complete observational CPU-only source/trace audit; not a new auth-policy intervention.**

## Reproducible real GitHub evidence

- [GitHub Actions **#44679395972**](https://github.com/kolihhan/small-agent-qlora/actions/runs/44679395972): frozen AppWorld source lock, two original Qwen task shards, safe aggregate **SUCCESS**. Authoritative safe aggregate artifact **14604192387**; per-scene **14604267028** (#1), **14603499528** (#2). All source/model digests checked.
- Frozen original model Qwen3.5-9B Q4 GGUF SHA256 `8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54`, original pinned AppWorld source `42b5bcf3cd334fee33f0c37c02070a9f5807add5`, source-frozen TRAIN scenes 1 and 2. 20 model steps, 384 max generated tokens per step, max 48 API calls, two original tool definitions. Model prompt/observations unchanged.
- Token provenance is inspected only inside the protected simulator: exact occurrence of the supplied `access_token` value in model-visible task instructions or an earlier successful API's public tool-result JSON `access_token` value, including successful same-app login. Export only allowlisted classification/booleans, **never secret values or hashes**, task text, API/app name, raw tool output or private evaluator data.
- Also measures login calls **after** first 401, not only before. An unmatched nonempty token would be labeled **source not identified**, never **fabricated**. However neither of these original repeat trajectories supplied a token at its first 401, so neither provides *positive* provenance evidence.

## Official result and distinct mechanisms

| Source scene | First401 step | Pre-first401 relevant API docs actually viewed? | Login API docs viewed before first401? | Supplied token at first401? | After-first401 login attempts / successes | Official task success | Original Qwen worker |
|---|---:|---|---|---|---|---|---|
| #1 | **3** | **No** | **No** | **No** | **8 attempted / 0 succeeded** | **Fail** | 853.38 s, 20 model turns, nested 1/7 |
| #2 | **8** | **Yes** | **Yes** | **No** | **0 attempted / 0 succeeded** | **Fail** | 483.95 s, 9 model turns, nested 2/8 |

- Scene #1: **12 actual appworld_call requester invocations and 12 HTTP_ERROR responses, 401×11 + 422×1, no duplicate calls or blocked tool results**. Eight of these API calls were source-matched to same-app `login` tool calls after the first401; **all eight actually reached the AppWorld request API and failed**. The anonymized aggregate does not map particular login attempts to HTTP401 vs HTTP422 status; do not assert the 422 was definitely a login parameter error.
- Scene #2: 2 API calls, status 401×1 + 422×1, first 401 step8. Had viewed **both** protected API and login docs, but never attempted login after 401 either; exited with normal final at model step9. Remaining 12 nominal model steps at first401; budget limit did not force this task to stop.
- Both official task success 0/2; **no new benchmark improvement**. No runtime `model_error` events in this repeat (typed classifier works in tests but does not retroactively diagnose prior `model_error` A/B).
- Sum of original worker execution time 1,337.33 s (~22.3 worker-minutes), not clock latency with both shards parallel. This source-observational follow-up should not be repeated on the same tasks without a new discriminating hypothesis.

## Causal interpretation and stop gate

**P0 partly identifies the failure path but does not establish usable credentials or a causal repair.** There are at least two competing mechanisms:

1. **Missed precondition/action choice:** scene2 knew there was a login API, lacked a token and never attempted login, despite 12 steps remaining at first401. An auth-before-protected-action planner could alter this branch, but only if actual login prerequisites are satisfied.
2. **Login execution failure:** scene1 originally lacked a token and only after HTTP401 invoked login **eight times**, all rejected by the public requester; simply moving login earlier may produce the same failures. Need login-specific **privacy-safe HTTP status/argument contract classification**, whether credentials were available from the task/tool and whether an actual token was ever returned before designing the A/B.

**A critical prior-run discrepancy:** The earlier original source Qwen diagnostic [#44675856600](https://github.com/kolihhan/small-agent-qlora/actions/runs/44675856600) saw scenes #1/#2 supply nonempty rejected tokens at first401, without prior login. The latest source-frozen reruns had **NO supplied token** at first401 on both. This is *repeated trajectory instability*, not evidence that the previous tokens were invented or are known to be valid. No raw prior trace was saved, so their provenance **cannot be recovered from this artifact**.

### Recommendation

- **Do NOT launch a generic auth-order A/B or QLoRA yet.** A future A/B requires verifying login inputs and public authorization error categories for the exact attempted app, without exporting credentials, checking not just whether login tool was attempted but whether it succeeded, and strict same-prefix/private-simulator-state matching. Primary metric remains **official AppWorld task success**, not fewer 401s or a successful login alone.
- Scene1 has a *login-call failure* hypothesis; scene2 has a *login-not-attempted* hypothesis. They should be tested as **distinct branches**, not aggregated as one policy weakness.
- Both source tasks were repeatedly exposed. A future generalization verdict must use new source-scene-disjoint tasks. No TRAIN→GAIA transfer has been established.
- No code changes to original agent policy, no production PR merge, no QLoRA. **The reference A/B from PR #99 remained 0/1 eligible official rescues** and adding the same auth message again is not evidence-based.

**Conclusion: P0 observational audit complete for the two frozen repeated scenes, but neither inferred/validated token origin of a NONEMPTY rejected token. New auth-order intervention remains gated.**
