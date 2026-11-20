# Official AppWorld Supervisor Parity — opted-in only; real CPU contract proof (2026-11-20)

**COMPLETE P0. The original two-tool AppWorld environment has an independently verified auth capability omission. We restored the two official Supervisor credential lookup APIs behind a literal boolean opt-in and verified real read-only responses in the private AppWorld simulator. NO fixed-Qwen A/B, task success, or training occurred.**

## Scope and exact upstream evidence

- Base original scaffold `AppWorldDocsTool` and `AppWorldCallTool` in `src/gaia_small_agent/tools/appworld_tool.py`; `_SUPERVISOR_ALLOW = {'complete_task'}` remains unchanged as the default.
- Official pinned upstream AppWorld revision **`42b5bcf3cd334fee33f0c37c02070a9f5807add5`** demonstrates the ordinary auth sequence `supervisor.show_profile()`, `supervisor.show_account_passwords()`, application `login(...)`, then access token. [Upstream public reference](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/prompts/full_code_agent/full_code_instructions.txt).
- Original model Qwen3.5-9B GGUF remains pinned at **`8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54`**, **not loaded** in any P0 test. The exact original 8 scene-distinct TRAIN source roster is the previously exposed pilot cohort, not a fresh promotion sample.
- Earlier [capability audit #44742581683](https://github.com/kolihhan/small-agent-qlora/actions/runs/44742581683) physically proved the standard public credential retrieval path is documented but blocked by the original bridge on **8/8** original scenes.

## Implementation: guarded, NOT an authentication shortcut

- `allowed_public_api(world,app,api,*,official_supervisor_parity=False)`: original `complete_task` stays callable; with **literal `True` only**, and only if the app is actually in the task allowlist and the API appears in the task's **public** docs, permit *exactly* `show_profile` and `show_account_passwords`. All other Supervisor APIs, `admin`, `api_docs`, private/underscore APIs remain denied. Non-Supervisor task allowlist unchanged.
- `AppWorldDocsTool(world,official_supervisor_parity=True)` discovers these two public API names and allows per-API public parameter documentation. A targeted **response shape** is derived from official AppWorld's public `response_schemas.success` **examples**, but replaces every example value with type metadata. It never shows fabricated password/email examples as if they were valid credentials. Both API docs fit the original tool-observation 2,000-character presentation.
- `AppWorldCallTool(world,official_supervisor_parity=True)` routes authorized calls through the original `world.requester.request(_app_name=...,_api_name=...,show=False,...)`; **never** performs automatic login, hardcodes credentials, provides shortcuts, or adds extra API access. No `AgentRuntime` / existing Qwen research script opts in unless explicitly configured. The original default-off behavior and existing tool definitions remain intact.
- The new documentation description is part of the **treatment input**. It is legitimate that baseline/treatment model-visible inputs differ at the first step. A paired future A/B must match original **task/DB initialization/model/budgets**, **not** demand identical tool-definition tokens across two genuinely different tool environments.

## Completed real CPU-only workflow evidence

[**Full pinned AppWorld Actions #44747327799**](https://github.com/kolihhan/small-agent-qlora/actions/runs/44747327799) **SUCCESS**, two immutable anonymized artifacts:

| Evidence | Artifact ID | Result |
|---|---:|---|
| Eight-scene opt-in ON vs original OFF public docs + method/path/parameter + 2,000-char response **shape** and allowlist | **14634242021** | 8/8 parity permits **two** additional Supervisor endpoints, original permits **zero**, both preserve completion; public docs shape is discoverable within original model cap |
| Eight-scene private simulator **real read-only** calls, two per scene; only booleans/counts uploaded | **14633637097** | **16/16 actual lookups successful**; top-level type and overlapping public-example field types compatible, returned JSON stays valid inside 2,000-char Agent observation for all 16; no login, state mutation, new model calls or task evaluator |

- The public response contract is generated from upstream public **examples**, not a comprehensive formal JSONSchema, so **top-level plus overlapping fields** does not prove every optional field/value invariant. The empirical real private shape and parseability is nevertheless confirmed.
- There is **one public schema profile** across eight scene-distinct tasks: do NOT count this as eight independent login implementations. All eight use the same AppWorld API definitions.
- All pytest synthetic CPU tests, original-tool regression tests and pinned actual source tests passed on the official workflow. The new redacted probe uses a private workspace and suppresses any SDK stdout/stderr, returns only allowlisted booleans/counts, and never writes actual task, profile, account password, access token, original response, or its hash to GitHub artifacts. Credentials may become **model-visible in the treatment** as legitimate tool outputs during a future Qwen experiment; that is required by the task, but future A/B runs must keep model traces and all payloads *inside the private runner*.

## Mechanism and future causal validation

**P0 validated:** The original 0/4 Task Success result remains the true original-system score but is confounded by a demonstrably missing *standard* credential retrieval affordance. Generic 401 error prompts or premature QLoRA do not fix an absent tool endpoint. Restoring this capability still does not establish that the model will ask for it, perform correct login, manage access tokens, or complete official tasks.

**P2 approval gate / pre-declared next experiment (not yet executed):**
1. Freeze a new **scene-disjoint** AppWorld task cohort; reserve prior TRAIN8 for development and report prospective choices before seeing performance. Avoid implicit leakage from this source-visible pilot.
2. Paired A/B from the same AppWorld task/initial DB, same fixed Qwen3.5-9B GGUF SHA, **20 turns / 384 output tokens per turn / 48 API calls**, same original prompt/system and requesters. **Only change** whether Supervisor parity is opt-in for both DocsTool and CallTool. Treatment tool description/docs necessarily differ from baseline; do NOT require identical first-step model input.
3. Export **only** safe counters (whether Supervisor profile/password lookups were attempted/succeeded; login attempt/success; HTTP status histogram; task success and nested state tests; cost/steps). Do not log account details, passwords, tokens, profile, task instruction, generated trace, or private evaluator content. Keep raw traces/values private. Any auth token present must come from an actual legitimate returned tool response, not a hardcoded value.
4. Primary endpoint: publisher-official **AppWorld Task Success**, with paired rescues/regressions and per-task state subtests. Login success and HTTP-error reductions are **intermediate diagnostics**, not benchmark improvement. Verify sensitivity to 2,000-char observations and twenty-turn budget.
5. No claim of GAIA transfer and **NO QLoRA/SQL training** until an honest causal improvement or a new true model-decision bottleneck survives matched A/B.

**Conclusion:** infrastructure parity repair passed both source-only and real private CPU contract checks. It is an eligible future controlled intervention, **not** an achieved agent success improvement. PR remains **DRAFT, DO NOT MERGE**.
