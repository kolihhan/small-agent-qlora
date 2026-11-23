# Branch map

This is a research repository with many historical experiment branches. A branch name is not a model version or evidence of a better score. Here is the short route to the code worth reading.

| Purpose | Branch / link | Why it exists |
| --- | --- | --- |
| Public project entry point | [main](https://github.com/kolihhan/small-agent-qlora/tree/main) | Stable CLI, tests, core agent, controlled v3D study and curated documentation |
| Portfolio documentation release | [release/portfolio-gaia-20261123](https://github.com/kolihhan/small-agent-qlora/tree/release/portfolio-gaia-20261123) | Human-readable README and verified result docs; documentation only |
| Current GAIA100 evidence | [exp/gaia-fourway-20261123](https://github.com/kolihhan/small-agent-qlora/tree/exp/gaia-fourway-20261123) | Actual 300 paired Base/v7/v8 Step70 episodes, source checks, score recovery and repeat workflow |
| Physically verified v8 Step70 training | [research/v8-stage70-resume-20261121](https://github.com/kolihhan/small-agent-qlora/tree/research/v8-stage70-resume-20261121) | Original QLoRA step35-to-step70 continuation and optimizer-state verification |
| v8 GGUF conversion evidence | [research/gaia-shadow25-v3b-transfer-proof-20261122](https://github.com/kolihhan/small-agent-qlora/tree/research/gaia-shadow25-v3b-transfer-proof-20261122) | Verified conversion workflow and GAIA tool-agent experiments |
| Earlier controlled models | [experiment/heldout-next-action-eval](https://github.com/kolihhan/small-agent-qlora/tree/experiment/heldout-next-action-eval) | Historical Base/v3B/v3D matched GAIA evidence and diagnostics |
| Unpromoted finalization candidate | [exp/gaia-fast-20261123](https://github.com/kolihhan/small-agent-qlora/tree/exp/gaia-fast-20261123) | Proposed terminal nudge; not a verified score improvement |

## What belongs where

- **main:** reviewed project code, installation instructions, reproducible methods and results with linked evidence.
- **release/**: short-lived presentation/docs branches to merge into main after CI.
- **exp/**: active controlled comparisons; never assume one is production-ready.
- **research/** and **experiment/**: exploratory or historical work, including negative results.
- **fix/** and **debug/**: point-in-time engineering investigations.

The November comparison is not a direct continuation of the October v3D result. In October, the same-run Base/v3B/v3D scores were 9/18/18 on the historical runtime. In November, the same-run Base/v7/v8 Step70 scores were 18/25/29 on a later runtime. Keep these tables separate.

## Branch cleanup policy

Preserve experimental history until the source checkpoint hashes, matching run logs and negative ablations are inventoried. Deleting or force-rebasing older branches can destroy the only surviving record of a run after GitHub Actions artifacts expire.

For a clean portfolio link, share **main**, not an individual experimental branch. For an exact result or implementation, follow the branch and run link in [the GAIA report](gaia-2026-11-23.md).

No experiment branches were deleted as part of the portfolio packaging.
