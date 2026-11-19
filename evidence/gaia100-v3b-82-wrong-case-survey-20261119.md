# GAIA100 v3B — 82 Wrong-Case Evidence Survey

This report uses the frozen Base/v3B/v3D *same-run* Evaluation100 from [GitHub Actions #38911211243](https://github.com/kolihhan/small-agent-qlora/actions/runs/38911211243) (2026-10-16). The complete aggregate JSON was recovered from the original summarizer job logs. The 20 archived per-task traces have expired and cannot be read through the current artifact API. **This cannot be a verified causal per-question 82-row survey.**

## 82 cases by mutually exclusive observed termination outcome

| Outcome | Wrong cases |
|---|---:|
| `invalid_final` | **51** |
| `model_error` | **3** |
| `model_timeout` | **2** |
| Final answer emitted but wrong | **26** |
| **Total** | **82** |

## By official GAIA difficulty (same source protocol)

| Level | Correct | Wrong | Total |
|---|---:|---:|---:|
| Level 1 | 6 | **26** | 32 |
| Level 2 | 10 | **42** | 52 |
| Level 3 | 2 | **14** | 16 |

## Prioritized existing automatic proxy labels (exactly one per wrong task, but not causal)

| Proxy label | Count | Interpretation / uncertainty |
|---|---:|---|
| `incomplete_other` | 31 | Did not complete; no higher-priority proxy. Not a reason why evidence was missing. |
| `duplicate_action_present` | 21 | Some duplicate actions were blocked; may coincide with `invalid_final`. |
| `incorrect_after_tools` | 13 | Final answer wrong after tools, under priority labeling. Content not archived. |
| `capability_gap` | 6 | Recognized unsupported image/audio/video attachment type; score zero. |
| `tool_error_present` | 6 | Errors present, after higher-priority buckets assigned; not all errors necessarily causal. |
| `premature_final_proxy` | 5 | Wrong answer without any tool calls; tool use may or may not be necessary. |

## Cross-cutting signal counts (OVERLAP, cannot add)

- 25 of 82 wrong had at least one duplicate call block.
- 23 of 82 wrong had at least one tool error.
- 6 unsupported-attachment capability gaps.
- 5 wrong finalized with zero tool calls.
- Across all 100 tasks: 784 tool calls, 681 tool successes, 37 tool errors, 66 duplicate blocks. Duplicate blocks are tracked separately, so these are not 784 independent successful/failed tool executions.

## Why is the problem bigger than stopping?

At least two sequential bottlenecks: (1) many runs fail to finalize (56/82). (2) another 26 run to a final answer but produce a wrong answer. For (1), a forced-final generator could increase completion yet still be wrong if supporting evidence was never retrieved. For (2), deeper question decomposition, evidence retrieval/verification, tool use on attachments, and short-answer extraction require per-task trace adjudication; no per-task causal breakdown can be supported from expired archives.

Frozen v3D comparator on *same tasks* also has 18 correct but only 25 finals and 69 invalid finals; the additional v3D tool calls/duplicate blocks show worsening stop behavior, not general proof that search causes failure. The later v7 run scores 26/100 in another protocol (74 wrong), and cannot be mixed with v3B 82 wrong as a single 100-row analysis.

## Low-regret optimization of unchanged runtime

1. Validate a stop/continue label only when the step-level source already delivered enough evidence; do not train from a stop reason alone.
2. For non-final cases, distinguish missing evidence from evidence already delivered; use actual `read` observations.
3. For wrong-final cases, record whether the answer entity is in observed evidence, whether multi-hop information was combined, and whether answer was only incorrectly formatted.
4. Track task outcome, completion, tool error, duplicate rate, and tool cost separately, with matched v3B controls.
5. For unsupported image/audio/video tasks, mark capability boundary; QLoRA alone cannot supply a missing modality parser.

**Evidence boundary:** The diagnostic is aggregate-based, not an individual 82-task causal audit. To genuinely survey all 82 one by one, the original per-task `results.jsonl` files must be recovered (or a new protocol-matched run performed with permission and an isolation plan).