# Candidate A — bounded tool-free finalization

Status: preregistered before candidate code.

## Observation

Historical inspectable Diagnostic evidence shows termination instability: some runs exhaust the tool-step budget or stop with an empty final even after observations exist. Historical first-13 results are DEV evidence only and are not used as a performance claim.

## Hypothesis

If terminal-state handling is the bottleneck, then allowing exactly one final model call with `tools=[]` after either (a) an empty final or (b) exhaustion of the normal tool-step budget will recover some otherwise incomplete answers without increasing tool calls.

## Treatment

One generic control-policy treatment: bounded tool-free finalization recovery.

- no change to model, tools, scorer, GAIA question text, search implementation, or normal step budget;
- recovery may run at most once for an empty final and once at the exhausted budget boundary, but no recovery call can execute a tool;
- if recovery still returns empty content or a tool call, preserve the original failure outcome;
- no case-specific GAIA logic.

## Predictions

Development prediction: incomplete termination decreases without additional tool execution.

Blind GAIA promotion criterion: Candidate A is a GAIA performance improvement only if Shadow25 exact correct count is at least baseline +2, completion regresses by no more than one task, and no new catastrophic runtime failure class appears.

Efficiency/completion improvement without the exact-correct gate is not a GAIA performance win.
