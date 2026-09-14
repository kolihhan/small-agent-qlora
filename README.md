# Small-Agent QLoRA

Work in progress: a local Qwen3.5-4B tool agent with search, file reading, inspection, and Python tools.

I'm using this repo to test a fairly simple question: can QLoRA help a small model choose and use tools more reliably, or does it just add training cost without much benefit?

There is no Base vs LoRA result yet.

## Agent loop

```text
user
  |
Qwen3.5-4B
  |
choose next action
  |---- search
  |---- read
  |---- inspect
  |---- python
  |
observation -> next action -> final answer
```

The model chooses what to do next. The harness runs the tool, returns the result, limits the number of steps, blocks exact duplicate calls, and records the visible action trace.

This is a single-model loop, not a planner/router/critic graph.

## Current status

The runtime, evaluation runner, trajectory collector, QLoRA training entry point, and Base-vs-LoRA comparator are implemented.

The current experiment has not finished yet. I have not trained an adapter for the final comparison, and I do not claim that QLoRA improves the agent.

An earlier diagnostic run stopped after 13 saved tasks because the machine hit the RAM guard. That partial run got 2/13 exact match and completed 6/13 tasks. I keep it for debugging history, not as a 25-task accuracy result. No adapter was trained from that run.

More detail on that attempt is in `docs/p4-decision.md`.

## Experiment plan

The GAIA validation set is split into separate development and evaluation partitions:

```text
GAIA validation: 165

Diagnostic:       25
Evaluation:      100
Unused:           40
```

The 25 diagnostic tasks are for finding failure patterns. Training tasks are then created separately and must not copy GAIA questions or answers. Only successful tool-use trajectories that pass the checks are used for training.

After the adapter is fixed, Base and +LoRA are run on the same 100-task evaluation split and compared task by task.

The split is an internal evaluation setup, not an official GAIA leaderboard score.

## Run the agent

Install the pieces you need:

```powershell
python -m pip install -e ".[eval,search,files,train]"
```

Run a question:

```powershell
small-agent run "your question here"
```

Run the diagnostic partition:

```powershell
small-agent gaia-eval `
  --partition diagnostic `
  --backend transformers `
  --work-root runs/gaia-diagnostic-base
```

Collect verified training trajectories:

```powershell
small-agent collect-trajectories `
  --tasks data/policy-tasks.jsonl `
  --output data/verified-trajectories.jsonl `
  --backend transformers `
  --protected-questions runs/gaia-protected-question-hashes.json
```

Train an adapter:

```powershell
small-agent train-qlora `
  --data data/verified-trajectories.jsonl `
  --output adapters/qwen3.5-4b-tool-policy `
  --protected-questions runs/gaia-protected-question-hashes.json
```

Then run Base and +LoRA on the evaluation split and compare them with `small-agent compare-evals`.

## What gets recorded

For each task the runner keeps the final answer, correctness, stop reason, steps, tool calls, tool errors, duplicate calls, and the visible action trace. It also separates tasks the current tools can reasonably handle from tasks that need semantic image/audio/video understanding.

That distinction matters because the current `inspect` tool can read document and tabular structure, but it is not a general vision or audio tool.

## Training-data guard

GAIA questions are hashed locally and checked against training inputs. Training also rejects sources marked as GAIA.

This catches exact reuse. It does not detect a paraphrase of a benchmark question, so training-data provenance still needs manual judgment.

## Repository shape

```text
small-agent-qlora/
├── src/gaia_small_agent/
│   ├── agent/
│   ├── model/
│   ├── tools/
│   ├── benchmark/
│   └── training/
├── docs/
└── tests/
```

## Limits

- The Python tool is process-isolated and time-limited, but it is not a hardened hostile-code sandbox.
- `inspect` does not provide semantic image/audio/video understanding.
- Search uses the live web, so exact results can change.
- Exact hash checks do not catch paraphrased benchmark leakage.
- The Transformers/QLoRA path needs a compatible local CUDA/PyTorch/bitsandbytes setup.

## References

- Pi minimal harness: https://pi.dev/
- Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
- GAIA dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
- GAIA scorer: https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py
