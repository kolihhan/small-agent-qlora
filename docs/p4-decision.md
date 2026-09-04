> **Historical v1 evidence only.** This document records the preserved pre-review-v2 stop decision. It is not the current QLoRA conclusion; review-v2 treats the QLoRA question as unanswered and runs a new bounded protocol without altering these artifacts.

# P4 decision: `NO_QLORA_NOT_JUSTIFIED__DIAGNOSTIC_INSUFFICIENT`

Status: **FROZEN**. Keep the Base agent. Do not train QLoRA, create an adapter,
or open/run `evaluation100` from this remediation cycle.

## Facts

- Frozen experiment: local Qwen3.5-4B Base agent on the existing GAIA
  Diagnostic25 partition, 25 cases (8/13/4 by level), seed `gaia-local-v2`,
  Transformers 5.16.1, 4-bit NF4/double quantization, no adapter, native
  default KV cache, thinking off, 12 steps, 512 new tokens, and exactly
  `search/read/inspect/python`.
- Model snapshot commit: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
  model identity SHA-256
  `b0d72c1a2210829a8cb8b62a7faa0d6db25e2ef6a02c210499c112876c422140`.
- Dataset revision: `682dd723ee1e1697e00360edccf2366dc8418dd9`.
  The sealed `diagnostic25.ids.json` SHA-256 is
  `beaae1621b8078cdc6726c268c210b3a2a61a03afef4c25093ba873a4e69340f`;
  the runtime manifest wrapper SHA-256 is
  `b515f9f63bdf014353db08a484627e7e1a5f5e35011338f82603cc8409f29a40`.
  These are different files; their 25 IDs and order match exactly. The sealed
  evaluation100 ID artifact remains unchanged at SHA-256
  `16c317353a536d16660ea80ef20bbc8398788c530898f14bef107a98e7ad7a98`.
- Consolidated preflight: PASS with no inference, 89 tests, compile exit 0,
  all four real tool smokes, exact model/context/dataset/config identities,
  no implementation-hash mismatch, gold isolation, no candidate training
  JSONL, sufficient launch resources, and absent output. SHA-256:
  `f2427a4d6acf7034a3f917d2ee6bf57d1516f972c330c449e72d3a1b88734d9d`.
- First attempt: infrastructure-invalid during model loading, before inference.
  It is preserved at
  `runs/p4-base-diagnostic-v1.invalid-resource-20260829-a`; its resource monitor
  SHA-256 is
  `97fe2b53ba968ff65f563fd451f40b03b40102e4dd5d5e872ca7b6e0d0d499d9`.
- One permitted rerun used the unchanged experiment and frozen resource gate.
  It stopped after 13 atomic records; case 14 had no persisted result and 12
  cases therefore have no record. Coverage is 8 Level 1, 5 Level 2, 0 Level 3.
  The stop was available RAM below 512 MiB for more than five seconds. Wall
  time was 2,342.107 s; minimum available RAM 459,726,848 bytes; peak GPU use
  7,887 MiB; minimum free GPU memory 62 MiB.
- Persisted-prefix metrics: exact match 2/13 (15.38%), completed 6/13 (46.15%),
  118 model calls, 112 tool calls, 91 tool successes, 21 tool errors, 13
  duplicate blocks, and 81.25% tool success. Mean/p50/p95 case latency was
  159.671/152.994/446.283 seconds. These are prefix facts, not a full
  Diagnostic25 score.
- Sol-level classification of the persisted prefix observed 2 correct cases
  and 11 supported policy failures: 7 `max_steps`, 2 incorrect finals after
  usable tools, 1 premature-final proxy, and 1 duplicate-action case. No
  persisted case was assigned a semantic capability gap or causal
  infrastructure failure. Because the overall diagnostic is invalid and
  incomplete, formal Gate A is not evaluated from this prefix.
- Gate B was frozen false before results: no licensed, provenance-complete,
  oracle-verified non-evaluation trajectory dataset exists. No training,
  adapter, or evaluation100 action occurred.
- Independent integrity audit passed for the evidence that exists: exact
  result-ID uniqueness/order prefix, config equality, implementation hashes,
  evaluator-only gold, independent exact-match recomputation with zero
  mismatches, tool surface, model/dataset provenance, and raw artifact hashes.
- Local execution made zero billed model API calls. Energy and hardware-dollar
  cost were not measured.

Canonical frozen verdict: `runs/p4-base-diagnostic-v1/verdict.json`, SHA-256
`9946eb3d64a98e0529c7dca3f8982291850c0b124ec000338dd3944c4830abc6`.
Raw persisted results SHA-256:
`c7c445147bacc782607d13fbb93b4c1d6dbcdf755729511c01b0b40d17049b02`.
Resource monitor SHA-256:
`5ab2f43124f55b3a611de9a6d90986dc60a2417ac212ba0692e5f587bf898782`.

## Interpretation

The local hardware did not support a valid complete run under the frozen
resource contract, so the partial results cannot establish Diagnostic25
accuracy or the prevalence of model-policy failures. Independently, Gate B
already makes QLoRA unjustified. The smallest honest outcome is to keep the
Base implementation, preserve the invalid evidence, and stop this remediation
cycle without another experiment.

Known limitations: no Level 3 result persisted; live-web evidence can change;
no Base-versus-QLoRA transition counts exist; no energy/dollar estimate was
measured; exact question hashes do not detect paraphrases; and the Python tool
is not a hardened hostile-code sandbox.
