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
- Base/+LoRA run-comparability checks;
- deterministic policy-task provenance and the rule that training prompts do not name the required tool.

The package exposes `small-agent run`, `gaia-eval` (`gaia100` alias), `collect-trajectories`, `train-qlora`, and `compare-evals`.

The frozen P4 preflight on 2026-08-29 recorded `89 passed` and compile success. That count belongs to the preserved experiment evidence; it is not a claim about the current repository test count. Current code health is reported by GitHub Actions so the documentation does not become stale every time a regression test is added.

The independent frozen-artifact audit also checks the exact 25-ID ordering, unique ordered result prefix, evaluator-only gold, deterministic score recomputation, run-config equality, implementation hashes, model/dataset identity, tool surface, resource samples, and unopened evaluation100.

The real local run is intentionally recorded as `INVALID_INCOMPLETE_RESOURCE_STOP`, not repaired into a complete benchmark: 13/25 results persisted before the sealed RAM guard fired. The 2/13 prefix is not a Diagnostic25 accuracy claim. The frozen verdict and raw hashes are in `runs/p4-base-diagnostic-v1/verdict.json`.
