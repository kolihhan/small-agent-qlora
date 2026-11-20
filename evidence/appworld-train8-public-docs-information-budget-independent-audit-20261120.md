# AppWorld TRAIN8 pinned real source: public API docs visibility, 2k cap and HTTP provenance (2026-11-20)

**COMPLETE SOURCE-ONLY DIAGNOSTIC. No Qwen inference, no GAIA, no adapter, no accuracy improvement.**

## Frozen upstream and exact provenance

- AppWorld source SHA `42b5bcf3cd334fee33f0c37c02070a9f5807add5`, same as both prior TRAIN4 real model studies.
- Pinned original Qwen3.5-9B GGUF SHA `8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54` **not loaded or called** by these source-only jobs.
- Real upstream source census [physical Actions #44655442042](https://github.com/kolihhan/small-agent-qlora/actions/runs/44655442042) **SUCCESS**, artifact **14592323643**. CPU-only; model calls 0; never reads protected reference evaluator or outputs task instructions, public API parameter names, actual default values, raw exceptions or private app names.
- Corresponding safe production-model-visible-error *unchanged* diagnostic counters and synthetic tests: `src/gaia_small_agent/tools/appworld_public_contract_audit.py`, `appworld_tool.py`, `tests/test_appworld_public_contract_audit.py`; Linux and Windows CI SUCCESS. Upstream Requester.raise_if_failure source at pinned commit raises precisely `Response status code is NNN:\n{response.text}`; original 47 HTTP errors did not preserve status in earlier artifacts, so **cannot reconstruct their 4xx/5xx distribution retroactively**.

## Actual public API documentation data — no multiplying duplicate scene catalogs

Eight task-scene source rows have **one distinct count profile after removing the scene index**. The 8 source rows are not 8 independent API catalogs or 8 independent statistical observations. One representative scene:

| Field | Actual one-scene count |
|---|---:|
| Public API docs inspected | **469** |
| Public parameter entries | **1,544** |
| Parameters beyond the Tool's first-24 cap | **0** |
| Public parameter `default` fields present but hidden | 1,544 |
| **Non-null** public defaults actually hidden | **363** |
| Public parameter `constraints` fields present but hidden | 1,544 |
| **Nonempty** public constraints actually hidden | **444** |
| Public API response-schema/example metadata present but hidden | **469** |
| Current API docs exceeding AgentRuntime **2,000-char** observation cap | **13/469** |
| Estimated enriched docs exceeding same 2,000-char cap | **17/469** |
| Extra API docs newly crossing the bound | **4/469** |
| Total current documented API-output characters (if all 469 inspected) | **300,548** |
| Total projected after displaying only non-null defaults/nonempty constraints | **325,071** |
| Increment | **24,523 chars = 8.16%** |

The last two size figures are **source-only estimates for all 469 doc lookups**; they are not actual per-agent token use, runtime or measured cost. Model tool responses additionally impose the fixed 2,000-character cap, which may truncate appended details. Response-schema/example metadata was counted as absent but **not** added to enriched-size projection (likely much larger).

## Independent challenge and stop-loss

1. **Verified missing public fields ≠ root cause of agent failure.** The student may fail because of planning, credentials, entity resolution, tool surface, absent session state, shortened generation (384 tokens/call) or 20-turn budget. No causal A/B exposure yet.
2. **The previous “24-parameter cap” concern is falsified on this source:** 0 parameters hidden by that cap. Stop treating it as an observed failure.
3. **The default/constraint key-presence count alone is misleading:** all 1,544 parameters include both keys upstream, but only 363 non-null defaults and 444 nonempty constraints; counts may overlap and must not be summed as unique parameter omissions.
4. **HTTP statuses from prior 47 failed API calls were not retained.** New safe counters record numeric HTTP status and public validation hint categories only, without raw exception text or modifying Qwen's observations. They are instrumentation for a future run, not retrospective measured status distributions.
5. **Documenting many APIs is not useful if the model cannot pick a correct one.** A passive availability audit does not prove an improvement in app/tool selection or official AppWorld state.
6. **New model-visible long docs might worsen performance:** 13/469 already clipped, projected 17/469 clipped; every extra tool call costs time and tokens. Improve one relevant API explanation on demand only if real model-visible pre/post comparison proves a specific blocked action.
7. **Identical count profiles are not identical task state.** This audit cannot establish that all public API docs or simulator state bytes are identical across eight scenes; only observed count summaries match.

**Decision:** keep production/default bridge responses unchanged; no bulk doc injection, no QLoRA. [Prior two TRAIN4 runs #44565392162](https://github.com/kolihhan/small-agent-qlora/actions/runs/44565392162) and [#44568286156](https://github.com/kolihhan/small-agent-qlora/actions/runs/44568286156) both had original 0/4, exact-prefix intervention denominator 0. Before spending CPU/GPU on any more original Qwen episodes, first pre-register a real observation/first-obstruction status data capture under a protected local runner (no raw export), with source-verified public fields, stable state and success criteria. Require nonzero trigger and official task-success rescue to claim effect.

**PR scope:** research-only source/doc/status counter observability. Source docs themselves are public; no protected instructions, answers or API secrets exported. Do not merge until useful.
