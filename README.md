# Small-Agent QLoRA

**A local Qwen3.5-4B tool agent that tests one question: does verified policy-trajectory QLoRA improve agent policy on a frozen GAIA evaluation partition?**

```text
user
  ↓
Qwen3.5-4B
  ↓
choose next action ── search / read / inspect / python
  ↑                         ↓
  └──────── observation ────┘
  ↓
final answer
```

The runtime is **Pi-inspired**, not built on or forked from Pi. The model chooses the next action; the harness executes tools, returns observations, enforces bounds, blocks exact duplicate calls, and records a visible trace. There is no planner/router/critic graph and no GAIA-specific action policy.

The portfolio claim is not that QLoRA must win. The experiment is designed so that Base vs +LoRA differs only by the adapter; if the adapter does not justify its cost, that negative result is the result.

## Current review-v2 experiment

The previous v1 diagnostic remains immutable evidence, but its `NO_QLORA_NOT_JUSTIFIED__DIAGNOSTIC_INSUFFICIENT` verdict is **historical, not the review-v2 conclusion**. Review-v2 fixes the text-only model path, tool isolation, extensionless file detection, trajectory admission, and the circular training-data gate. The new experiment has not been run yet; `scripts/run_p4_full_pipeline.ps1` is the canonical restart-safe pipeline. It requires at least 32/64 clean verified training trajectories with read/inspect/python coverage before QLoRA may start.

### Historical v1 remediation outcome

P4 was frozen at `NO_QLORA_NOT_JUSTIFIED__DIAGNOSTIC_INSUFFICIENT`. The
consolidated preflight passed, but the unchanged local Base Diagnostic25 hit
the frozen host-RAM guard after 13 persisted records (8 Level 1, 5 Level 2,
0 Level 3). That prefix scored 2/13 exact match with 6/13 completed; it is not
a 25-case accuracy result. A preceding permitted attempt stopped during model
loading before inference. No adapter was trained and `evaluation100` was not
opened or run because the independently sourced training-data gate was already
false. See [docs/p4-decision.md](docs/p4-decision.md) for hashes and limitations.

## Local experiment workflow

This project is intentionally **local-only for experimentation**; all benchmark and QLoRA runs are designed to execute on the local machine.

Install the runtime/evaluation/training dependencies:

```powershell
python -m pip install -e ".[eval,search,files,train]"
$env:HF_TOKEN = "<your Hugging Face token>"
```

GAIA remains supporting evaluation evidence, not training data or the product identity.

### 1. Diagnostic run — Base only

Run the 25-task diagnostic partition first:

```powershell
small-agent gaia-eval `
  --partition diagnostic `
  --backend transformers `
  --work-root runs/gaia-diagnostic-base
```

The deterministic diagnostic quotas are:

```text
Level 1   8
Level 2  13
Level 3   4
Total    25
```

Use this partition for failure analysis. The runner records transparent diagnostic labels such as `premature_final_proxy`, `tool_error_present`, `duplicate_action_present`, `max_steps`, and known semantic media capability gaps. These labels are **rule-based diagnostics, not causal root-cause claims**.

Do **not** use the later 100-task evaluation partition to decide training data or hyperparameters.

### 2. Build independent non-GAIA training tasks

Create tasks that exercise the failure classes seen on diagnostic cases without copying GAIA questions, answers, or trajectories:

```json
{"id":"recovery-001","source":"synthetic-recovery","question":"...","expected_answer":"..."}
```

Every `gaia-eval` run writes a local exact-match protection file at:

```text
runs/gaia-protected-question-hashes.json
```

It stores normalized SHA-256 question hashes for the full GAIA validation pool, not gated question text. Both trajectory collection and training reject:

- sources containing `gaia`;
- exact normalized question overlap with the protected hash set.

This guard prevents direct/exact reuse. It is **not** a semantic paraphrase detector, so the training-data provenance still has to be reviewed honestly.

Collect only **policy-verified successful trajectories**:

```powershell
small-agent collect-trajectories `
  --tasks data/policy-tasks.jsonl `
  --output data/verified-trajectories.jsonl `
  --backend transformers `
  --protected-questions runs/gaia-protected-question-hashes.json
```

### 3. Train and freeze the adapter

```powershell
small-agent train-qlora `
  --data data/verified-trajectories.jsonl `
  --output adapters/qwen3.5-4b-tool-policy `
  --protected-questions runs/gaia-protected-question-hashes.json
```

Training is turn-level `state -> next assistant action` supervision. The 4-bit base stays frozen and LoRA is applied directly to the text-only causal language model; the multimodal parent/vision tower is not loaded.

Freeze the adapter and experiment settings before running the final comparison.

### 4. Frozen local evaluation — Base and +LoRA

The 100-task evaluation partition is deterministic and disjoint from the 25 diagnostic tasks:

```text
Level 1  32
Level 2  52
Level 3  16
Total   100
```

Run Base **after the adapter/config is frozen**:

```powershell
small-agent gaia-eval `
  --partition evaluation `
  --backend transformers `
  --work-root runs/gaia-evaluation-base
```

Then run the same model/harness/config with the adapter:

```powershell
small-agent gaia-eval `
  --partition evaluation `
  --backend transformers `
  --adapter adapters/qwen3.5-4b-tool-policy `
  --work-root runs/gaia-evaluation-qlora
```

The evaluation runner checkpoints `results.jsonl`, validates the local manifest and run config on resume, and skips already-persisted tasks.

### 5. Paired comparison

```powershell
small-agent compare-evals `
  --base runs/gaia-evaluation-base `
  --adapter-run runs/gaia-evaluation-qlora `
  --output runs/gaia-evaluation-comparison.json
```

The comparator refuses the comparison unless:

- the same task IDs are present;
- Base has no adapter;
- E1 has an adapter;
- every recorded run option except `adapter` is identical.

It reports:

```text
Base accuracy
+LoRA accuracy
wrong -> correct
correct -> wrong
net correct change
average steps
average tool calls
average tool errors
supported-subset recovery/regression
```

## Why capability-aware reporting matters

The current generic file tools can extract text/structure from common text, PDF, CSV and XLSX artifacts, but they do not claim semantic image/audio/video understanding. The runner therefore separately reports:

```text
all selected tasks
capability-supported tasks
known semantic media capability-gap tasks
```

A failure caused by missing image/audio/video semantics should not be presented as evidence that the agent policy itself is bad or that QLoRA should fix it.

## Experiment split

The local protocol uses one fixed SHA-256 ranking within each GAIA level:

```text
GAIA validation: 165

Diagnostic:       25   ← development / failure analysis
Evaluation:      100   ← frozen Base vs +LoRA comparison
Unused:           40
```

The partitions are deterministic and disjoint. The GAIA dataset revision is pinned in code.

This evaluation pool has historical exposure in this project, so the README does **not** call it a pristine untouched benchmark. The correct claim is: the evaluation partition is frozen from this protocol onward and is not used to design the current intervention.

**This remains an internal split of GAIA validation, not an official GAIA leaderboard score.** Live-web questions can also change over time.

## What is measured

Per task, the runner persists:

- final answer and GAIA correctness;
- completion / stop reason;
- steps;
- tool calls / successes / errors;
- duplicate calls blocked;
- visible action/tool trace;
- known capability gap;
- rule-based failure label.

Summary output includes accuracy by GAIA level, completion/tool statistics, supported-vs-gap populations, and failure-label counts.

An empty model response with no tool call is **not** counted as completed; it stops as `empty_final`.

## Agent runtime

```python
while steps < max_steps:
    turn = model(messages, tools)
    if turn.has_tool_calls:
        observations = execute(turn.tool_calls)
        messages += observations
    else:
        return turn.final_answer
```

The harness does **not** encode `search -> read -> python -> answer`. Tool errors become structured observations so the model can choose a recovery action.

## CLI

```text
small-agent run <question>
small-agent gaia-eval --partition diagnostic|evaluation
small-agent collect-trajectories --tasks FILE --output FILE
small-agent train-qlora --data FILE --output DIR
small-agent compare-evals --base DIR --adapter-run DIR
```

`gaia100` remains an alias for `gaia-eval` for compatibility, but new experiment documentation uses `gaia-eval`.

Useful runtime flags:

```text
--backend ollama|transformers
--model qwen3.5:4b
--hf-model Qwen/Qwen3.5-4B
--adapter PATH
--max-steps 12
--max-new-tokens 512
--thinking
```

For controlled Base vs +LoRA evaluation, use the Transformers backend for both arms.

## Repository shape

```text
small-agent-qlora/
├── README.md
├── src/gaia_small_agent/
│   ├── agent/                     # minimal autonomous loop
│   ├── model/                     # Ollama + HF/adapter backends
│   ├── tools/                     # search/read/inspect/python
│   ├── benchmark/                 # local partitions, runner, scorer, comparator
│   └── training/                  # protection, verified trajectories, QLoRA
├── docs/
└── tests/
```

## Status

Implemented and locally testable without GAIA data/model weights:

- Pi-inspired autonomous single-model/tool loop;
- visible action trace;
- structured tool errors and exact-duplicate guard;
- empty-final failure handling;
- four generic tools;
- Ollama product/smoke backend;
- Qwen3.5 Transformers backend with optional LoRA adapter;
- deterministic disjoint diagnostic/evaluation GAIA protocol;
- local checkpoint/resume validation;
- exact-content GAIA training-contamination guard;
- verified non-GAIA trajectory collector;
- QLoRA training entry point;
- Base/+LoRA paired comparator;
- supported-vs-capability-gap reporting.

The only frozen real-GPU result is the explicitly invalid 13-record diagnostic
prefix above. No Diagnostic25 accuracy, Base-versus-QLoRA comparison, or QLoRA
improvement is claimed.

## Limitations

- `python` is process-isolated and time-bounded, not a hardened hostile-code sandbox.
- `inspect` currently handles document/tabular structure and metadata; semantic image/audio/video understanding is not implemented.
- Search uses live web results, so exact runs can vary.
- Exact question-hash protection does not detect paraphrased benchmark leakage.
- The Transformers/QLoRA path requires a local CUDA/PyTorch/bitsandbytes setup compatible with the machine.

## References

- Pi minimal harness: https://pi.dev/
- Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
- GAIA gated dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
- GAIA official scorer: https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py
