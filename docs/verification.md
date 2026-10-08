# Verification

Repository verification is intentionally separated from real benchmark evidence.

Current CPU-only GitHub Actions verifies the supported package/runtime surface on both Linux and Windows with Python 3.11. Each matrix job:

- installs the editable package with the development and file-reader extras;
- checks that the `small-agent` CLI entry point loads;
- runs the full pytest suite;
- compiles `src` and `tests` with `compileall`.

The current test suite covers:

- autonomous loop / tool recovery / duplicate blocking;
- structured model timeout, unavailable-backend, capacity, and model-error stops;
- bounded remote and workspace-file reads;
- empty-final handling;
- Qwen protocol parsing;
- file and Python tool safety boundaries;
- non-destructive runtime readiness checks;
- single-run artifact provenance;
- GAIA scoring and deterministic diagnostic/evaluation partitioning;
- checkpoint/resume consistency;
- capability-gap/failure-label summaries;
- exact-content GAIA training protection;
- trajectory and QLoRA data contracts;
- Base/+LoRA run-comparability checks.

The package exposes `small-agent doctor`, `run`, `gaia-eval` (`gaia100` alias), `collect-trajectories`, `train-qlora`, and `compare-evals`.

The frozen P4 experiment verification recorded on 2026-08-29 is `89 passed`; `compileall` exited 0. That number belongs to the preserved experiment artifact and is not the current repository test count. Current repository verification should be read from GitHub Actions rather than copied into this document as a moving test-count claim.

Passing repository CI does **not** establish model quality, GAIA accuracy, or a QLoRA improvement. Those claims require the separate frozen local experiment contract.

The independent artifact audit for the preserved experiment checks the exact frozen 25-ID ordering, unique ordered result prefix, evaluator-only gold, a deterministic score recomputation, run-config equality, implementation hashes, model/dataset identity, tool surface, resource samples, and unopened evaluation100.

The real local run is intentionally recorded as `INVALID_INCOMPLETE_RESOURCE_STOP`, not repaired into a complete benchmark: 13/25 results persisted before the sealed RAM guard fired. The 2/13 prefix is not a Diagnostic25 accuracy claim. The frozen verdict and raw hashes are in `runs/p4-base-diagnostic-v1/verdict.json`.
