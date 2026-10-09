# SDD ledger — plan: docs/superpowers/plans/2026-10-09-gaia-improvement-framework.md

Execution method: Native / PIC
Implementation branch: gaia/improvement-framework-v1

Pre-flight interfaces:
- Task 1 shadow selector -> Task 3 CLI and Task 4 workflow: `partition="shadow"` through existing runner.
- Task 2 promotion helper -> Task 4 workflow: aggregate summary dictionaries only.
- Tasks 1–5 framework baseline -> Task 6 clean candidate branch: candidate must contain exactly one behavioral treatment.

Ruling: GitHub connector execution replaces local Superpowers worktree scripts; branch isolation + CI/TDD provide the same safety boundary available in this harness.
