# Reproduce the project

There are two levels of reproduction here: lightweight runtime checks, and the pinned model-plus-adapter experiments. The second requires model files, GPU/CPU time and gated benchmark access. The first runs locally without downloading a 9B model.

## 1. Package and tests

Use Python 3.11+.

~~~bash
git clone https://github.com/kolihhan/small-agent-qlora.git
cd small-agent-qlora
python -m pip install -e ".[dev,search,files]"
small-agent --help
python -m pytest -q
~~~

The GitHub Actions [CI workflow](../.github/workflows/ci.yml) installs the package, runs pytest and checks compilation on Linux and Windows. Passing CI checks code behavior; it does not validate GAIA accuracy.

## 2. Local agent smoke

Start an Ollama server and pull an available Qwen3.5 model:

~~~bash
ollama pull qwen3.5:9b
small-agent doctor --backend ollama --model qwen3.5:9b
small-agent run "Find the original Transformer paper and give its title." \
  --backend ollama --model qwen3.5:9b \
  --workspace runs/demo --output runs/demo/result.json
~~~

The doctor checks the tools and runtime without serving as an accuracy benchmark. The CLI prints tool-call summaries and writes a JSON trace. Network-backed search may need internet access.

The CLI defaults to Ollama. The reported GAIA100 evaluation did **not** use that default; it used a fixed local llama.cpp server exposing the OpenAI-compatible chat API.

## 3. Frozen GAIA100 experiment

Use the [frozen experiment branch](https://github.com/kolihhan/small-agent-qlora/tree/b63fa3b088ff608458933531387a3be49ba33049) to inspect the actual workflow and scorer contract.

The workflow performs:

1. Verify search, read, inspect and restricted Python on the runner.
2. Load GAIA validation at pinned revision 682dd723ee1e1697e00360edccf2366dc8418dd9 using HF_TOKEN.
3. Select the exact Evaluation100 task IDs with gaia-local-v2; check level quotas and full task-ID checksum.
4. Verify the base GGUF, llama.cpp release and original adapter files with SHA256.
5. For each of 20 shards, run Base, v7 and v8 Step70 **sequentially on that shard's same runner**.
6. Save task hashes, correct/completed flags, stop reasons, tool errors and paired wins/losses. The benchmark reference answers remain private to the evaluator.

The full experiment is computationally expensive. If you only want to inspect the result, download [the saved summary](https://github.com/kolihhan/small-agent-qlora/actions/runs/45173486537/artifacts/14818152995) instead of triggering another GAIA run.

Original run: [45144537782](https://github.com/kolihhan/small-agent-qlora/actions/runs/45144537782). Its evaluation jobs succeeded; its aggregator lacked a package install. Recovered summary: [45173486537](https://github.com/kolihhan/small-agent-qlora/actions/runs/45173486537), which reused all saved inference outputs without changing the scores.

Reported protocol: Qwen3.5-9B Q4_K_M; 16,384 context; 12 agent steps; max 512 output tokens/call; temperature 0; reasoning budget 0; tool observation ceiling 2,000 chars.

### What you cannot reproduce from a source clone alone

The repository does not include several-gigabyte base weights, adapter binaries or gated GAIA dataset content. Old training/conversion artifact links can expire. The v3D GGUF was not recovered for the November four-way comparison and must not be silently replaced. Reproducing the original run requires the *same* checkpoint bytes; a new training run with similar settings is a different model.

## 4. Controlled policy study

The 200-task synthetic closed-loop study and v3D continuation live in the stable research code and [its original report](v3d-results.md). These are separate from the later real-tool GAIA100 experiment. Training code and generation/verification scripts are in [src/gaia_small_agent/training/](../src/gaia_small_agent/training/) and [scripts/](../scripts/).

The project rejects exact GAIA question overlap in training data, but it does not make a claim about every possible paraphrase or indirect leak. Treat GAIA100 as a previously seen research benchmark, not an unseen test set.

## 5. Troubleshooting

- **Doctor fails at search:** check internet access and the installed search extra; the live search provider can be unavailable.
- **Ollama connection refused:** start Ollama and confirm the model tag is installed.
- **No HF_TOKEN:** GAIA validation access needs a token with the appropriate dataset access.
- **Missing adapter artifact:** stop. Locate the original file and verify its SHA256; do not swap in another checkpoint.
- **Final answer is invalid:** the model may emit tool-call markup as plain text even when structured tool calling is disabled. The runtime records this separately from a wrong normal answer.
- **Mismatch with a historical score:** first check dataset task IDs, runtime commit, base GGUF hash, adapter hash and live tool conditions. Scores from different runtime revisions are not directly matched evidence.

For related runtime contracts, see [verification](verification.md) and [architecture](architecture.md).
