# Final matched GAIA v2 ledger

- Base: **12/100 (12.0%)**
- QLoRA: **8/100 (8.0%)**
- Delta: **-4.0 pp** (-4 correct)
- Completion: 31.0% -> 12.0%
- Tool errors: 355 -> 467
- Duplicate calls blocked: 105 -> 223
- Paired gains/regressions: 1/5; exact McNemar p=0.2188
- Verdict: **reject_current_adapter_for_promotion**

Hard gates: 100 unique task IDs per arm, identical task-ID sets, level mix 32/52/16, frozen selection hash matched, zero invalid attempts in final ledger.
