# GAIA fast v3B terminal-nudge candidate — 2026-11-23

## Read-only protocol comparison

Sources:
- v3B same-run GAIA100 workflow: main/.github/workflows/gaia-v3d-clean-same-runner.yml
- v3B runtime: 08fbcaa5ed82c5238188c45d5748993bae9bf1e1
- v7 paired protocol: experiment/v7-format-gaia100-replication-20261116
  (.github/workflows/qlora-v7-format-gaia100.yml, scripts/gaia_v7_format_eval100.py)
- v7 recorded outcome: Actions 43865621132, v7 original 26/100 vs format reminder 26/100.

| Dimension | v3B matched Base/v3B/v3D GAIA | v7 paired GAIA100 |
|---|---|---|
| Base | twoody Qwen3.5-9B Q4_K_M, SHA 8c4cc24f... | Same exact GGUF SHA |
| Adapter | qwen35-9b-qlora-v3b GGUF, SHA e7caec60... | different trained v7 F16 GGUF, SHA 475f5fb5... |
| Steps | 12 | 12 |
| Final step | existing OpenAICompatModel.complete_final with tool_choice=none; fallback no tools | same |
| Unexpected final tool call | still rejected by AgentRuntime | same |
| Tools | search, read, inspect, python | same names AND same implementation blob SHA across all four |
| Observation bound | 2,000 characters | 2,000 characters |
| Agent SYSTEM_PROMPT | identical literal | identical literal |
| Benchmark question appended format | identical (runner.py same SHA) | identical |
| Scorer / Evaluation100 partition | same scorer and partition source SHA | same |
| Model runtime | llama.cpp b11541, --ctx-size 16384 --jinja, --reasoning-budget 0, --temperature 0 | same |
| Max completion tokens | 512 | 512 |
| Model request timeout | 180s | 180s |
| Prompt ablation | none | optional system reminder: output just code/number/value, no explanation |
| Actual format-ablation outcome | n/a | 26/100 original, 26/100 reminder, 8 wins/8 losses |
| Runtime protocol code | earlier OpenAICompatModel rejects malformed JSON args with model_error and does not preserve reasoning_content | later decoder passes reasoning_content and turns invalid JSON arguments into sentinel to permit tool correction; AgentRuntime carries reasoning_content in transcript |
| Runner organization | 20 five-task shards; Base -> v3B -> v3D on each runner, each with own llama-server lifecycle | 20 five-task shards; same v7 adapter for both prompt arms, same llama-server per shard |
| Physical tool return stabilization | none; live tool randomness | none; live tool randomness |
| Training relationship | v3B checkpoint itself | different v7 checkpoint; no attribution of v7 score to protocol alone |

No demonstrated net GAIA accuracy benefit from v7's format reminder. We do not port it.
The 26% score from the later v7 run cannot be compared causally to historical v3B 18% because the adapter and runtime differ.

## Candidate (one isolated intervention)

This branch starts from main and modifies AgentRuntime with an optional
finalization_nudge flag, default False. Baseline is the unchanged behavior.
Candidate enables the flag only; at step 11 (max_steps-1) it appends an
ephemeral final-answer-only reminder to the existing system prompt. At step 12
it repeats the reminder while invoking the already existing complete_final /
tool_choice=none path. Does not modify the stored message transcript.
No new calls, training, dataset, tools, or scorer changes.
If an API returns unexpected tool_calls despite tool_choice=none, existing
max_steps failure logic remains; this intervention cannot guarantee an answer.

## Evaluation

Workflow: GAIA Fast v3B terminal nudge paired Evaluation100
File: .github/workflows/gaia-fast-v3b-paired.yml

Run uses pinned base + exact historic v3B adapter SHA256
e7caec609a905d21e056b107519d78916d3ef79e7bdc70db916e40d801d2187d.
Each of 20 shard runners evaluates five same tasks: baseline then candidate
on a single llama-server with the same original v3B adapter, model and tools.
The 100-task identity SHA is
869e24fd53a0eeabc2598d952d0b5f893eca279584c96eb407a72906dccba45a.
Temperature=0, reasoning-budget=0, max_steps=12, max_new_tokens=512.
The tool/search environment is live and may vary across sequential episodes.
This is a fast single paired screen, NOT a 3-run or held-out validation.

Each shard exports only hashed task IDs, correctness, completion, stop reason,
invalid_final, max_steps, tool metrics and latency. GAIA question text, gold
answers, raw generated answers and traces remain temporary on the runner and
are not uploaded. Aggregate insists on 20 complete shards and 100 paired unique
tasks. Final JSON/Markdown include all 100 outcomes and win/loss case tables.

## Actual execution blocker and required manual setup

GitHub Actions run 38156235327 (original v3B converted GGUF) currently exposes
zero downloadable artifacts; its original v3B SHA is known, but bytes are not
recoverable from this old workflow. Run 43434228699 still has a v7 GGUF artifact,
but using it would NOT satisfy the original v3B requirement. No substitution.

The workflow first attempts the original run-id artifact. If unavailable,
set workflow_dispatch input v3b_adapter_url to an authenticated-accessible
direct HTTPS GGUF file URL containing the exact original v3B bytes. SHA256
validation is mandatory. If unavailable it fails BEFORE downloading the base
model, with ORIGINAL_V3B_GGUF_MISSING. HF_TOKEN must be configured as a
repository Actions secret to access the GAIA dataset.

Note: GitHub's manual workflow UI requires the workflow_dispatch YAML to be
present on the default branch to be listed/dispatchable. This experiment is
committed only on its isolated branch. Review and merge the workflow onto
the default branch before attempting the Actions UI dropdown, then choose
exp/gaia-fast-20261123 in Run workflow if that ref is still available.
No workflow was dispatched or model accuracy measured when preparing this branch.

## Post-run artifact

gaia-fast-v3b-paired-complete:
- gaia-fast-v3b-paired-summary.json (all task hashes, scores, win/loss rows)
- gaia-fast-v3b-paired-report.md (correct and invalid_final counts, wins/losses table)

No promotion decision until score is physically observed.
