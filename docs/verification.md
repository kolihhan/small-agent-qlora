# Verification

Local verification is intentionally separated from real benchmark evidence.

Unit tests cover:

- autonomous loop / tool recovery / duplicate blocking;
- empty-final handling;
- Qwen protocol parsing;
- file and Python tool safety boundaries;
- GAIA scoring and deterministic diagnostic/evaluation partitioning;
- checkpoint/resume consistency;
- capability-gap/failure-label summaries;
- exact-content GAIA training protection;
- trajectory and QLoRA data contracts;
- Base/+LoRA run-comparability checks.

The package exposes `small-agent run`, `gaia-eval` (`gaia100` alias), `collect-trajectories`, `train-qlora`, and `compare-evals`.

The frozen P4 experiment verification recorded on 2026-08-29 is `89 passed`; `compileall` exited 0. That number belongs to the preserved experiment artifact and is not the current repository test count.

Current repository CI has since expanded the test suite. The latest successful `main` run before this remediation passed `110` tests; subsequent changes should use GitHub Actions as the current verification source rather than rewriting the historical P4 record.

The independent artifact audit also checks the exact frozen 25-ID ordering,
unique ordered result prefix, evaluator-only gold, a deterministic score
recomputation, run-config equality, implementation hashes, model/dataset
identity, tool surface, resource samples, and unopened evaluation100.

The real local run is intentionally recorded as
`INVALID_INCOMPLETE_RESOURCE_STOP`, not repaired into a complete benchmark:
13/25 results persisted before the sealed RAM guard fired. The 2/13 prefix is
not a Diagnostic25 accuracy claim. The frozen verdict and raw hashes are in
`runs/p4-base-diagnostic-v1/verdict.json`.
