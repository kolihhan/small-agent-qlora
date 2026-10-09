# Candidate A — bounded finalization recovery

Status: HYPOTHESIS

Baseline SHA: `5bd335d5f68c0449ac3a728a1d28110f74f468b1`

Hypothesis: bounded tool-free finalization recovery reduces incomplete termination without increasing tool use; it counts as a GAIA performance improvement only if blind Shadow exact-correct improves by at least 2 tasks.

Treatment: one tool-free recovery path covering step-limit and empty-final termination. The recovery call receives `tools=[]`; no recovery tool call is executed. No planner, critic, router, extra search policy, or GAIA-case-specific prompt content is added.

Predeclared Shadow gate:

- exact correct delta >= +2;
- completion delta >= -1;
- no new catastrophic runtime stop class.

Efficiency/completion improvements without the exact-correct gate do not count as a GAIA performance win.
