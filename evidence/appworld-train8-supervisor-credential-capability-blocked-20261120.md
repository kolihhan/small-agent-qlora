# AppWorld original Qwen scaffold: official credential acquisition API physically blocked by bridge (2026-11-20)

**COMPLETE CPU-only evidence. This is a genuine task-environment capability problem, NOT a measured model capability failure or new task improvement.**

## Grounded source and actual-run evidence

- **Pinned AppWorld upstream** source commit `42b5bcf3cd334fee33f0c37c02070a9f5807add5` ([official code-agent authentication example](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/prompts/full_code_agent/full_code_instructions.txt)) explicitly obtains login credentials with `apis.supervisor.show_profile()` and `apis.supervisor.show_account_passwords()`, then calls `apis.<app>.login(username=...,password=...)` to get an access_token. The [upstream function-calling prompt](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/prompts/function_calling_agent/instructions.txt) explains that supervisor personal information/credentials are stored in the Supervisor app and retrieved through its APIs.
- **Our actual student bridge** `src/gaia_small_agent/tools/appworld_tool.py` hard-codes `_SUPERVISOR_ALLOW = {'complete_task'}`, denies **every other** supervisor endpoint via `allowed_public_api()`, and `AppWorldDocsTool.run({'app':'supervisor'})` advertises **only complete_task**. The runtime prompt `prompt_for(world)` in `scripts/appworld_train_first_feedback_fork.py` includes only the user's task instruction plus generic directions, not the Supervisor account profile or a retrieval alternative.
- **Real original frozen AppWorld TRAIN8 CPU-only Actions [#44742581683](https://github.com/kolihhan/small-agent-qlora/actions/runs/44742581683)** ALL SUCCESS; immutable safe output artifact **14631746968**, `APPWORLD-TRAIN8-SUPERVISOR-CREDENTIAL-LOOKUP-BRIDGE-GATE-SOURCE-ONLY`. No Qwen loaded, no login or other actual API call executed, no task prompt/profile, credential value, identifier or hash exported.
- Source and model ID locks unchanged, eight *previously exposed* TRAIN scene-distinct tasks; the public API catalog profile is identical across eight, so **NOT eight independent API implementations**.

## Exact real public schema and bridge findings

| CPU-only public/runtime question | Result across original 8 TRAIN scenes |
|---|---|
| Official `supervisor.show_profile` **and** `supervisor.show_account_passwords` in task public API docs | **8/8 YES** |
| Current AppWorld tool bridge permits both official credential APIs | **0/8** |
| Current bounded `appworld_docs` lists either credential API to original Qwen | **0/8** |
| Current `appworld_call` permits `supervisor.complete_task` | **8/8 YES** |
| Generic original prompt includes the task instruction | **8/8 YES** |
| Generic original prompt already contains exact supervisor account email | **0/8** |
| Public applications with documented login per scene | **9** |
| Sum of required login parameters for those 9 login interfaces | **18** |
| Distinct public schema count profiles over 8 tasks | **1** |
| Whether legitimate credentials happen to be retrievable from another allowed app or are independently supplied in task text | **UNDETERMINED** |
| Whether credentials would be valid for a particular target app / its permissions | **UNDETERMINED** |

### Causal interpretation

**Confirmed broken prerequisite:** The official standard credential retrieval path exists in the pinned world but is denied *by the experimental two-tool bridge*. This is an **environment/tool affordance confound** for the observed 401s, and is a stronger explanation than a generic model failure to plan login, though the original individual login failures remain not causally attributable from anonymized historical traces. It does NOT establish that each of the 8 tasks requires logging in, that every login is impossible by other legal routes, or that restoring these endpoints alone increases official Task Success.

Prior source-only [#44667209288](https://github.com/kolihhan/small-agent-qlora/actions/runs/44667209288) showed 372/372 required-token fields visible in public per-API docs **when the model requests those docs**. That result never proved *obtaining credentials* was possible. The first-401 original feedback study had only one valid prefix pair and 0/1 rescues; plain 401 hints would not fix the missing supervisor capability.

### Decision table for future intervention (strict stop-loss)

1. **No more Qwen auth-message/QLoRA experiments against the structurally credential-incomplete original bridge.** That conflates an environment bug with model capability.
2. First create a separate **opt-in**, source-only safety-tested *official credential lookup parity* treatment: expose exactly `supervisor.show_profile` and `supervisor.show_account_passwords` through the task-scoped public-doc/API bridge while retaining `complete_task`; no other supervisor/admin APIs. Keep original tool protocol, original 20 turns/384 tokens and same legitimate AppWorld data handling. Credentials and profiles must remain private to the task runner and model tool context, never uploaded to Actions logs or evidence artifacts.
3. Before any new costly Qwen, verify CPU-only API eligibility and public-contract exposure; the existing frozen scenes only qualify as an **exploratory pilot**. For a scored A/B use original fixed Qwen3.5-9B and a **new source-disjoint task** (prefer distinct scenario families), same model/call/budget and paired source state. Compare official AppWorld task success, login-step success, regressions, and cost; don't treat reduced 401 or a newly accessible API as an achieved task rescue.
4. If standard credential path is accessible but Qwen still fails to retrieve/use it, that is a much cleaner candidate for **Tool-grounded Decision Making** or targeted QLoRA. Training remains forbidden before this evidence.

**Result:** task environment is missing an intended legitimate capability; separate it from policy learning. No original model run, benchmark uplift, hidden task reading or unauthorized login occurred here. Keep PR DRAFT / DO NOT MERGE.
