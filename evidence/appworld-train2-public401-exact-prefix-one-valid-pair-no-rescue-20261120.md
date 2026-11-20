# AppWorld original fixed Qwen9B: complete first-401 single-public-auth-hint matched A/B (2026-11-20)

**RESULT: 1 strictly matched pair; 0 official rescues; 0 promotions.**

## Physical provenance

- [GitHub Actions #44662846202](https://github.com/kolihhan/small-agent-qlora/actions/runs/44662846202), source lock, both original scene shards and aggregate **SUCCESS**. Authoritative aggregate artifact `14597622401`; individually verified original/treatment scene-0 artifact `14597522427`; scene3 artifact `14596666490`.
- Fixed original Qwen3.5-9B GGUF SHA `8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54`, original pinned upstream AppWorld SHA `42b5bcf3cd334fee33f0c37c02070a9f5807add5`, no adapter, no GAIA.
- Frozen 20 model turns / 384 max generated tokens per call / 48 API call budget. Original source TRAIN tasks already previously seen by researchers (not prospectively independent).
- Only first tool-error 401 with a documented `access_token` contract qualifies. Replays exact prior Qwen model-visible input/decisions and validates *persisted private AppWorld DB checkpoint byte identity* at error step; changes only one public-auth feedback string. Private DB, task, access token, raw response never exported. This does not cryptographically prove all non-DB runtime state is identical.

## Verified per-case result

| Scene | Eligibility | Original official task success | Matched treatment official task success | Nested state test |
|---|---|---|---|---|
| 0 | **YES**, first HTTP401 required `access_token` absent, exact prefix + same private persisted DB checkpoint | False | False | **1/2 → 1/2** |
| 3 | **NO**, no qualifying first HTTP401 in this repeat original trace | False | not run | not applicable |

- **Valid paired denominator 1/2**. **Paired original 0/1, treatment 0/1; official task rescues 0, regressions 0** (no inference of zero population treatment efficacy).
- Scene 0 original completed with a regular final response after 17 model calls, 7 HTTP401 errors, while matched treatment used **11 new autonomous Qwen calls** and ended with **`model_error`**, still failed official goal.
- The `model_error` subtype was **not logged** in the published safe aggregate. `src/gaia_small_agent/model/openai_compat.py` maps HTTP/connection request errors, invalid JSON/choices, or malformed tool arguments into `model_error`; do NOT assert context length overflow or decode failure without typed evidence.
- Scene 3 original independently ended at step5, no first qualifying HTTP401. Repeated trajectories differ despite temperature 0; never count a nonexposed task as an unsuccessful treatment pair.

## Interpretation & stop loss

1. The specific idea "tell Qwen the required `access_token` is absent → it will recover and finish" **did not work on the sole valid paired task**. More 401 hint repeats on the same scenes without new mechanism would be adaptive selection and wasteful.
2. Historical original Qwen TRAIN4 observational run [#44660003397](https://github.com/kolihhan/small-agent-qlora/actions/runs/44660003397) measured **401×22 + 422×2** API failures, **official 0/4**, 2/4 forced terminal step20; 401 is strong observed symptom, but authorization failure could be missing credential, wrong permissions, token expiration or unperformed login.
3. The task's public API docs already specify `access_token` through the original bridge, so a repeated statement of required auth may be redundant. We need source-safe evidence whether the student actually looked up the relevant API docs/login and whether it ever acquired a valid token. Do not assume loss of public auth docs caused failure.
4. Before any further Qwen runs: instrument privacy-safe process-local status: prior *relevant API docs actually viewed*, presence of required token field in **delivered** docs observation, prior *login API called*, prior *login API succeeded*, whether token is absent/present (NOT token value), and model_error numeric HTTP/parse category. Do not publish API names/user task/credentials.
5. Compare *actual official task goal completion*, with pre-registered same prefix/private DB checkpoint and budget control; not just reduced 401s or a successful login.
6. **No QLoRA** based on this sample. Default tool behavior remains unchanged. At most a new narrowly isolated auth-planning mechanism after diagnostic evidence; not a blind batch of hints.

**Decision: record 0/1 matched official task success; stop unconditional auth hint.** No model switch, training or production merge.
