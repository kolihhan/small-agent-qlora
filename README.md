# Small Agent QLoRA

[![CI](https://github.com/kolihhan/small-agent-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/kolihhan/small-agent-qlora/actions/workflows/ci.yml)

**Can QLoRA teach a 9B model to use tools and finish real tasks more reliably?**

I trained Qwen3.5-9B adapters on tool-use trajectories, then tested them with an autonomous agent: search the web, read documents, inspect files, run limited Python, and decide when to answer. The interesting part was not the training loss. It was what happened when the model had to work through an unfamiliar task without a teacher telling it the next action.

The repo keeps the results that worked and the experiments that didn't.

## Results at a glance

Two different evaluations are reported below. They use different task sets and must not be mixed.

### Controlled tool-use policy test (200 tasks)

The earlier Base/v3B/v3D study used the same frozen deterministic tools and 12-step runtime.

| Model | Correct | Exact task success |
| --- | ---: | ---: |
| Base | 69 / 200 | 34.5% |
| v3B | 105 / 200 | 52.5% |
| v3D | **124 / 200** | **62.0%** |

v3D improved mostly at deciding *not* to call a tool when it already had the answer. It still failed the dedicated recovery cases. See [controlled results](docs/v3d-results.md) and [original Actions run](https://github.com/kolihhan/small-agent-qlora/actions/runs/38752732871).

### Real-tool GAIA100 comparison (one matched run, November 23, 2026)

The later experiment ran each available model against the **same 100 GAIA validation tasks**, using the same Qwen3.5-9B Q4_K_M base model, llama.cpp server, prompt, four tools, evaluator, temperature 0, and 12-step budget.

| Model | Correct / 100 | Finished with an answer | Invalid final | Tool errors |
| --- | ---: | ---: | ---: | ---: |
| Base | 18 | 47 | 38 | 38 |
| v7 | 25 | 37 | 50 | 36 |
| **v8 Step70** | **29** | **56** | **29** | **30** |

v8 Step70 fixed **14** Base mistakes and regressed on **3** Base-correct tasks: a net gain of **11/100**. It also made fewer tool calls (546 versus 724 for Base). The v7-to-v8 difference is smaller (25 versus 29) and is not yet established as a stable improvement.

The original v3D GGUF artifact had expired, so v3D was **not** evaluated in this particular run. Don't substitute its older 18/100 score into this table: the historical comparison used a different runtime revision.

[Read the matched run and limitations](docs/gaia-2026-11-23.md) · [Original 300 episodes](https://github.com/kolihhan/small-agent-qlora/actions/runs/45144537782) · [Recovered scorecard](https://github.com/kolihhan/small-agent-qlora/actions/runs/45173486537)

**Important:** this is a previously used GAIA100 research subset, not an untouched benchmark or a public leaderboard result. Additional repeat runs are tracked separately; no unverified repeat score is included here.

## How the agent works

There is one model and one tool loop, not a planner/critic ensemble.

~~~mermaid
flowchart LR
    Q[Question] --> M[Qwen3.5-9B + optional LoRA]
    M --> D{Next action}
    D -->|Search / Read / Inspect / Python| T[Tool execution]
    T --> O[Bounded observation]
    O --> M
    D -->|Answer| F[Final answer]
    M -->|Step limit or error| E[Recorded failure]
~~~

The runtime validates tool arguments, blocks exact duplicate calls, limits observations to 2,000 characters, and records the final answer, stop reason, tool errors, and per-step traces. Image/audio/video understanding is not part of the current tool set.

## Try it locally

Requires Python 3.11+ and a model server. Ollama is the simplest path for a local demonstration; the reported GAIA numbers used a pinned llama.cpp server instead.

~~~bash
git clone https://github.com/kolihhan/small-agent-qlora.git
cd small-agent-qlora
python -m pip install -e ".[dev,search,files]"
ollama pull qwen3.5:9b
small-agent doctor --backend ollama --model qwen3.5:9b
small-agent run "Find the author of the original paper on transformers." \
  --model qwen3.5:9b --output runs/demo/result.json
~~~

The doctor checks the local runtime and tools. A passed doctor check is **not** a model accuracy result. To run without a model or GPU, use the CPU test suite:

~~~bash
python -m pytest -q
~~~

For reproducible evaluation, adapter provenance, required Hugging Face access, and exact benchmark settings, see [Reproduction guide](docs/reproduce.md). A local Ollama run is not equivalent to the published GGUF/llama.cpp protocol.

## What I learned

- **Training for one behavior can hurt another.** v3D was stronger on the controlled policy tasks but often kept searching until it ran out of steps on GAIA.
- **Finishing is not the same as getting it right.** v8 Step70 emitted 56 final answers but only 29 were correct. Incorrect answers and missing evidence still matter even when the tool loop works.
- **Paired tests matter.** I tracked which individual tasks were fixed and which regressed. Comparing scores from different runner revisions can give the wrong impression.

The next research question is whether the model can judge when retrieved evidence is sufficient. It hasn't been solved in this repo.

## Repository map

| Path | What it contains |
| --- | --- |
| [src/gaia_small_agent/](src/gaia_small_agent/) | Agent loop, model adapters, tools, scoring and QLoRA training |
| [tests/](tests/) | Runtime, tool, data-leakage and evaluation-contract tests |
| [scripts/](scripts/) | Evaluation and audit entry points |
| [docs/reproduce.md](docs/reproduce.md) | Setup and experiment reproduction |
| [docs/gaia-2026-11-23.md](docs/gaia-2026-11-23.md) | Latest verified GAIA100 comparison and failure breakdown |
| [docs/v3d-results.md](docs/v3d-results.md) | Controlled study and negative ablations |
| [docs/branches.md](docs/branches.md) | Where the research code, evidence and release branch live |
| [.github/workflows/](.github/workflows/) | CI and preserved research workflows |

Many experiments live on separate Git branches. The default branch is the readable project entry point; experiments are preserved rather than folded blindly into production code.

## Limits

The tool runtime is a research prototype, not a secure multi-tenant sandbox. Its Python tool is restricted but should not be treated as a security boundary for untrusted code. Live web search can vary between runs. Some GAIA questions require modalities the agent doesn't support, and an exact-answer score doesn't explain every causal failure.

The controlled 200-task evaluation is synthetic. The GAIA100 study is real-tool evaluation on a **previously seen subset**. Neither result proves generalization to entirely new benchmarks. Model checkpoints are large and not shipped in this Git repository; experiment branches and Actions logs record their provenance and availability.
