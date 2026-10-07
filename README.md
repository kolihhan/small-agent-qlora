<h1 align="center">Small-Agent QLoRA</h1>

<p align="center">
  <strong>Can a local 4B tool agent learn a better tool-use policy with QLoRA — without hiding behind a larger model or benchmark leakage?</strong>
</p>

## At a glance

| | |
|---|---|
| **Question** | Can QLoRA improve the tool-use policy of a small local agent, or does it mostly add training cost? |
| **What I built** | A single-model Qwen3.5-4B agent with `search`, `read`, `inspect`, and `python`, plus verified trajectory collection, QLoRA training, leakage guards, resource guards, and a paired Base-vs-LoRA comparator. |
| **Current answer** | **Unanswered.** The experiment infrastructure exists, but there is no valid Base-vs-LoRA improvement claim yet. |
| **Design focus** | Controlled evaluation, visible trajectories, bounded tool use, and honest stop conditions under local hardware constraints. |

> [!NOTE]
> “QLoRA was not justified by the available evidence” is a valid outcome. This repository does not force a fine-tuning success story.

## The experiment

```text
GAIA Diagnostic25 (development evidence only)
        ↓
pre-registered non-GAIA local-tool curriculum
        ↓
verified clean trajectories
        ↓
QLoRA adapter
        ↓
sealed GAIA Evaluation100
   Base           +LoRA
      \           /
       paired comparison
```

The final comparison keeps the base model, tool surface, step/token limits, evaluation partition, and scorer fixed; only the adapter changes.

The Diagnostic25 partition is useful for describing observed failure modes, but **it does not generate or tune the training curriculum**. The synthetic curriculum is frozen independently so benchmark failures cannot be copied into training by construction.

## Agent loop

```text
user task
   ↓
Qwen3.5-4B
   ↓
choose next action
   ├─ search
   ├─ read
   ├─ inspect
   └─ python
   ↓
observation
   ↓
next action or final answer
```

The harness executes the tool, returns bounded observations, limits steps, blocks exact duplicate calls, and records the visible action trace. This is a single-model loop, not a planner/router/critic graph.

## Training curriculum

The current synthetic curriculum is deterministic, non-GAIA, and local-only. Its prompts describe the task **without naming the required tool**; the required tool is grader metadata used only to decide whether a trajectory is eligible for training.

Examples ask for a fact from a named file, file structure metadata, or a calculation large enough that using the bounded Python tool is reasonable. Only clean, answer-correct trajectories that used the required capability are kept.

`search` remains part of the runtime and GAIA evaluation surface, but it is **not directly trained by this deterministic curriculum** because live-web results would make the training set time-dependent. Generalization from local-tool training to web search is therefore an open question, not an established claim.

## Current evidence

An earlier diagnostic attempt stopped after **13 persisted tasks** because the machine crossed the RAM safety guard:

| Partial diagnostic fact | Result |
|---|---:|
| Persisted tasks | 13 / 25 |
| Exact match | 2 / 13 |
| Completed tasks | 6 / 13 |
| Model calls | 118 |
| Tool calls | 112 |
| Tool success | 81.25% |
| Stop reason | available RAM below guard threshold |

These are prefix/debugging facts, not Diagnostic25 accuracy and not evidence that QLoRA helps. No adapter was trained from that frozen run. See [`docs/p4-decision.md`](docs/p4-decision.md).

## Evaluation contract

GAIA validation is deterministically split into:

```text
Diagnostic:   25
Evaluation:  100
Unused:       40
```

The sealed Evaluation100 is not opened for model-selection decisions. Training rejects GAIA-labelled sources and exact normalized overlaps with protected GAIA questions; the hash guard does not detect paraphrases, so provenance still matters.

This is an internal controlled evaluation setup, not an official GAIA leaderboard submission.

## Canonical experiment path

The strongest reproducibility contract is the full pipeline script, not the individual CLI commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_p4_full_pipeline.ps1 `
  -RunRoot runs\p4-full-v2
```

That path pins the model/dataset revisions, runs offline where required, monitors resources, requires a complete Diagnostic25, requires at least 32 verified non-GAIA trajectories with local-tool coverage, requires both Evaluation100 arms to finish 100/100, and only then compares Base vs +LoRA.

### Development / individual stages

```powershell
python -m pip install -e ".[eval,search,files,train]"
small-agent run "your question here"
small-agent gaia-eval --partition diagnostic --backend transformers --work-root runs/gaia-diagnostic-base
small-agent generate-policy-tasks --output data/policy-tasks.jsonl --count 64
small-agent collect-trajectories --tasks data/policy-tasks.jsonl --output data/verified-trajectories.jsonl --backend transformers --protected-questions runs/gaia-protected-question-hashes.json
small-agent train-qlora --data data/verified-trajectories.jsonl --output adapters/qwen3.5-4b-tool-policy --protected-questions runs/gaia-protected-question-hashes.json
```

`compare-evals` is a low-level comparator. The canonical full pipeline is what enforces the two complete 100-case evaluation arms before comparison.

## Limits

- No valid Base-vs-LoRA result exists yet.
- The Python tool is process-isolated and time-limited, but not a hardened hostile-code sandbox.
- `inspect` does not provide semantic image/audio/video understanding.
- Live web search is non-deterministic and is not part of the synthetic training curriculum.
- Exact hash guards do not catch paraphrased benchmark leakage.
- The Transformers/QLoRA path needs compatible local CUDA/PyTorch/bitsandbytes.
- Hardware resource limits already prevented one complete frozen diagnostic run; partial prefixes are not promoted into benchmark results.

## References

- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- [GAIA scorer](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
