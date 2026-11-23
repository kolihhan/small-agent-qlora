# Verification guide

The repository has two independent verification layers. They answer different questions.

## Code checks

[CI](../.github/workflows/ci.yml) runs on pushes and pull requests on Linux and Windows, using Python 3.11. It installs the package, exercises the CLI, runs pytest, then compiles the source and tests.

~~~bash
python -m pip install -e ".[dev,search,files]"
small-agent --help
python -m pytest -q
python -m compileall -q src tests
~~~

These checks guard against broken tool contracts and runtime behavior. They do not evaluate model quality.

## Real tool readiness

Before the November matched GAIA100 inference, GitHub Actions checked the actual tool surface (search, read, inspect, restricted Python) and validated the fixed GAIA task identities. The preflight passed. This shows the tools *worked at the start of the run*; it does not establish that every later web query found the right evidence.

The [original comparison run](https://github.com/kolihhan/small-agent-qlora/actions/runs/45144537782) finished all 20 inference shards. Its summary job failed because the job did not install the Python package. The [recovery job](https://github.com/kolihhan/small-agent-qlora/actions/runs/45173486537) used the saved original shard artifacts to generate the verified scores, without re-running model inference. Both states are preserved instead of hiding the error.

## Score verification

Accept a GAIA score only when all of these hold:

- Same pinned GAIA dataset revision and task IDs for every arm.
- Same Qwen base weights and llama.cpp runtime with verified hashes.
- Same tool surface, prompt, per-step budget, generation settings, and scoring code.
- Original GGUF adapter verified; unknown or expired checkpoints must not be substituted.
- Exactly 100 unique scored task records per model; no missing or duplicated cases.
- Correctness, completion, finalization failures, tool errors and per-task wins/losses reported separately.

The latest verified matched result is documented in [GAIA results](gaia-2026-11-23.md). The historical Base/v3B/v3D study used a separate runtime and has its own [protocol and caveats](gaia-v3d-results.md).

## Old P4 experiments

Some research notes in this repository describe an earlier Qwen3.5-4B host-RAM-limited P4 diagnostic. That attempt stopped after 13/25 task records and was not a complete GAIA score. It is preserved as negative engineering evidence, not used to support the 9B results.
