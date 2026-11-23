# Archived QLoRA research code (November 2026)

These files are literal source snapshots from completed or in-progress experiment branches. They are preserved for reproducibility and audit. They do **not** run automatically from this directory.

- \`gaia_same_task.py\`: matching GAIA Evaluation100 task selection, hashed task outputs and comparisons, copied from the November experiment branch.
- \`gaia_v8_stability.py\`: repeat aggregation, copied as research code; its repeat run was separate from the one verified initial result.
- \`workflows/\`: original evaluation, result-recovery and repeat GitHub Actions YAML, copied as inert files.
- \`colab_v8_stage70_remote.py\`: Colab QLoRA continuation from verified step35 to step70.
- \`prepare_qlora_v8_multisource.py\`: source-balanced, non-GAIA training dataset preparation.

The *actual executed* GitHub Actions run IDs remain 45144537782 (initial paired inference), 45173486537 (score recovery), and 45175648911 (stability repeat workflow). The first two support the published 18/25/29 GAIA100 numbers; the repeat is not counted until a complete verified report is available.

For the precise original branch commits, refer to [historical heads](../../docs/archive/branch-heads-2026-11-23.json) and [branch history](../../docs/branches.md). Those original commits are preserved as ancestors of main, even after removing the old branch refs.

**Do not mix an archival script with the current main runtime and claim reproduction.** Use the original commit, model GGUF SHA, adapter SHA and benchmark revision specified in [the reproduction guide](../../docs/reproduce.md).
