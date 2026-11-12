# QLoRA stopping generalization: preregistered Shadow25 check

## Hypothesis and alternatives

Observed matched GAIA Evaluation100: v3B and v3D both answered 18/100 correctly,
but v3D finalized only 25/100 versus v3B 44/100. v3D's 69 invalid finals
were at the 12-step ceiling while attempting further tool calls.

**Primary hypothesis:** the state-aware 24-example layer-31 v3D continuation
learned short-horizon decisions but failed to generalize stopping to longer
tool trajectories.

**Not yet excluded:** backend/tool-call serialization; final-only tool_choice
handling; extra tool-search tendencies; missing evidence; model quantization;
small-sample variability. The current runtime already requests no tools on the
last step: simply adding tool_choice=none is NOT a new remedy.

## One-shot confirmatory diagnostic (no new training)

- Use deterministic GAIA shadow partition (GAIA_REVISION pinned);
  Shadow25 is disjoint from the earlier Evaluation100 and Diagnostic25 by
  select_gaia_partition.
- Exactly **25 tasks** / five shards of five; v3B and v3D sequentially on the
  same runner per shard, using same base GGUF, fixed adapters and hashes,
  temperature 0, reasoning budget 0, 16,384-token context, 12 steps,
  and 512 output tokens per model call.
- **Primary metric:** paired finalization difference (v3D vs v3B), with exact
  McNemar p-value, and per-arm invalid_final at step 12.
- **Secondary metrics:** paired local correctness, duplicate calls blocked,
  total tool calls, error counts; finalization by v3B-observed step buckets.
  The step-bucket comparison is descriptive and NOT a causal finding.
- Report all failures, not just positive results. The frozen Evaluation100
  must not be used to choose a winner after seeing this new Shadow25 result.
- No new adapter or PPO training. Do NOT infer final correctness from a
  textual tool-call formatting fix; do not claim official GAIA pass rates.
- This spends the Shadow25 holdout. Subsequent optimization must use
  independently generated tasks or a genuinely new reserved evaluation set.

## Privacy boundary

The paired workflow keeps protected GAIA questions, task IDs, answers,
and per-task traces **inside ephemeral runner workspaces only**. The ONLY
uploaded artifacts are per-shard aggregate metrics and a single aggregate-only
25-task summary. The CLI prints no protected rows.

Local usage with authorized protected matched result files:
python -m scripts.analyze_termination_shift --v3b /private/v3b/results.jsonl --v3d /private/v3d/results.jsonl --expected 100 --output /private/termination-summary.json

CI/tests use synthetic, non-GAIA records; a green CI does **not** mean the
actual Shadow25 model experiment passed.

## Decision rule

1. If v3D continues to finalize substantially less often on Shadow25, treat
   v3D as a targeted synthetic-policy checkpoint, NOT a general GAIA upgrade;
   keep v3B as the more defensible external checkpoint.
2. If Shadow25 reverses the stopping trend, report uncertainty (25 tasks
   is small) and do not claim that stopping is solved.
3. If both checkpoints fail to finalize, separately test the model transport
   with controlled tool-call-turn states. A backend parser change is a runtime
   ablation, **not** evidence of improved LoRA reasoning.
4. Only then build fresh synthetic long-horizon data with variable true
   stopping points and error-recovery states. Compare frozen untreated
   vs corrected checkpoints, with explicit cost and a new untouched holdout.

No metric should be silently promoted until its full model run and artifacts
are verifiable.
