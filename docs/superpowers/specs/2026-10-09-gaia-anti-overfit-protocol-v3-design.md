# GAIA Anti-Overfit Protocol v3

## Goal

Measure whether Small-Agent improvements generalize on unseen GAIA tasks instead of repeatedly optimizing against the same exposed first-13 diagnostic slice.

## Problem

The current workflow has repeatedly inspected and tuned against the same `Diagnostic25` prefix using `--limit 13`. This slice is not a representative mini-benchmark: `select_gaia_partition()` sorts the diagnostic partition by level, then the runner applies `selected[:limit]`, so the first 13 tasks are biased toward the beginning of the level-ordered diagnostic partition. Re-running fixes against this exposed slice risks benchmark overfitting.

The existing repository already has the important foundation: deterministic, disjoint `diagnostic` (25 tasks: 8/13/4 by GAIA level) and `evaluation` (100 tasks: 32/52/16) partitions at a pinned GAIA revision. v3 should reuse that design rather than introduce a new evaluation framework.

## Protocol

### 1. Diagnostic25 is the only development GAIA set

- Run all 25 diagnostic tasks when evaluating a candidate for development.
- Diagnostic questions, traces, answers, and failure labels may be inspected.
- Diagnostic25 may guide generic fixes.
- The historical first-13 runs remain development evidence only and must not be presented as GAIA benchmark improvement.
- `--limit` remains a debug convenience only; documentation/workflows must not use a limited run as a performance claim.

### 2. Add one deterministic blind Shadow partition

Use GAIA validation rows that belong to neither Diagnostic25 nor Evaluation100.

- Select deterministically from the remaining rows using a new seed namespace derived from the existing protocol seed.
- Preserve level coverage while fitting the actual remaining population. The implementation must first assert that the requested quotas are available.
- Preferred target is 25 tasks with the same 8/13/4 level quotas as Diagnostic25. If the pinned dataset revision cannot support those quotas after reserving Diagnostic25 and Evaluation100, fail loudly rather than silently changing the protocol.
- Shadow task IDs are persisted in a local manifest, but gated question text, reference answers, and traces are never committed or uploaded.
- Public CI output/artifacts may contain only aggregate metrics and protocol metadata.

### 3. Candidate lock before Shadow

A candidate is frozen before its Shadow run.

- Candidate SHA, model identity/digest, GAIA revision, partition seed, max steps, max tokens, thinking mode, and tool surface are recorded.
- After Shadow results are observed, changing behavior creates a new candidate generation. The same Shadow set must not become a tuning set.
- Do not inspect Shadow case-level traces/questions/answers during development.

### 4. Shadow promotion gate

Compare the frozen candidate with the frozen production baseline in the same GitHub Actions job, using the same Ollama server/model digest and the same Shadow manifest.

Primary gate:

- candidate must improve exact GAIA correct count by at least **+2 tasks** over baseline on Shadow;
- candidate completion count must not regress by more than **1 task**;
- candidate must not introduce a new catastrophic runtime failure class.

Secondary metrics (tool calls, tool success, duplicates, latency) are diagnostic only. Better efficiency without better exact correctness is a runtime improvement, not a GAIA performance improvement.

If the gate fails, stop that policy direction. Do not inspect Shadow task details and patch toward them.

### 5. Evaluation100 remains final

Only a Shadow-passing, frozen candidate may be promoted to the existing 100-task evaluation partition.

- Evaluation100 remains unseen during policy development.
- Run frozen baseline and candidate under matched configuration.
- This is the pre-QLoRA Base result used by the research story.

### 6. QLoRA remains isolated from GAIA content

- Training data comes only from verified non-GAIA trajectories / independent curriculum.
- GAIA diagnostic, Shadow, and Evaluation100 questions, answers, traces, or derived case-specific rules must not enter QLoRA training data.
- After training, compare Base vs +LoRA on the same frozen Evaluation100 manifest.

## Minimal Implementation

Ponytail rule: change the existing GAIA partition/runner machinery; do not add another evaluation framework.

Expected code surface:

1. `benchmark/gaia100.py`
   - extend deterministic partition selection with `shadow`;
   - reserve Diagnostic25 and Evaluation100 first, then select Shadow from the remainder;
   - keep manifest validation deterministic and disjoint.

2. `cli.py`
   - allow `--partition shadow`.

3. Tests
   - assert diagnostic/evaluation/shadow are pairwise disjoint;
   - assert exact quotas and determinism;
   - assert insufficient remaining per-level population fails loudly;
   - assert manifests reject partition mismatch.

4. One GitHub workflow
   - paired baseline/candidate Shadow run;
   - same runner/model server;
   - aggregate-only artifact;
   - computes the promotion gate without exposing case-level gated data.

5. Documentation
   - mark `--limit` as debug-only;
   - distinguish DEV Diagnostic25, blind Shadow, and final Evaluation100.

## Explicit Non-Goals

- No new planner, critic, router, memory layer, retry framework, or multi-agent architecture.
- No tuning against individual Shadow or Evaluation100 cases.
- No extra synthetic benchmark.
- No claim that completion-rate improvements are GAIA accuracy improvements.
- No QLoRA changes until the Base policy protocol is settled.
- No web-record/replay infrastructure unless matched-run evidence later shows live-search variance invalidates conclusions.

## Success Criteria

The protocol is complete when:

1. Diagnostic25, Shadow25, and Evaluation100 are deterministic and pairwise disjoint at the pinned GAIA revision.
2. CI can run baseline and one frozen candidate on Shadow in the same job without publishing gated case-level content.
3. The workflow emits a single aggregate promotion verdict based on the predeclared +2-correct / completion-regression gate.
4. A failing Shadow candidate is rejected without inspecting Shadow case details.
5. Only a passing candidate proceeds to Evaluation100.
