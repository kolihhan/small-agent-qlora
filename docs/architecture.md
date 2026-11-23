# Architecture

The agent is intentionally small: one language model selects a tool or writes an answer. The harness handles tool execution and safety checks; there is no planner, router or critic model.

~~~mermaid
flowchart TD
    U[Task + optional workspace file] --> L[AgentRuntime]
    L --> M[Qwen model: Base or LoRA]
    M --> A{Assistant turn}
    A -->|Tool call| T[Search / Read / Inspect / Python]
    T --> V[Argument validation, bounded result]
    V --> L
    A -->|Final answer| S[Record output and stop reason]
    L -->|Step limit or runtime error| F[Record failure]
~~~

## Inference and tools

- [Agent loop](../src/gaia_small_agent/agent/loop.py) owns the step budget, duplicate-call checks, tool failures, and final-answer handling.
- [Model backends](../src/gaia_small_agent/model/) include Ollama, direct Transformers and OpenAI-compatible servers. The November GAIA run used llama.cpp behind the OpenAI-compatible interface.
- [Tools](../src/gaia_small_agent/tools/) are search (live web), read (public URLs or workspace data), inspect (file metadata), and restricted Python. Search results may vary; the file tools intentionally limit remote reads.
- A tool observation is capped at 2,000 characters. The runner stores trace events and explicit stop reasons such as final, invalid_final, model_error and model_timeout.
- The Python helper is restricted, import-free, and subprocess bounded. It is **not** a general-purpose untrusted-code sandbox.

The training/evaluation model used for the reported QLoRA experiments is **Qwen3.5-9B**. Some older local diagnostic material in this repository used Qwen3.5-4B; those runs are not the reported 9B comparisons.

## Evaluation contracts

Two task families have different interpretations:

1. **Controlled policy 200:** frozen synthetic tasks and deterministic tools. Base/v3B/v3D are compared with the same runtime. This measures the targeted behavior of the supervised policy, not unrestricted real-world agent success.
2. **GAIA100:** 100 fixed GAIA 2023 validation tasks (32 Level 1, 52 Level 2, 16 Level 3). The November 23, 2026 comparison used real search/read/inspect/Python, fixed model and adapter hashes, and a single runner per five-task shard for each compared model.

GAIA task selection uses SHA-256 ranks and a fixed seed (gaia-local-v2). The Evaluation100 subset is distinct from the Diagnostic25 subset but has been seen by this project already. See [the complete November protocol](gaia-2026-11-23.md).

## Data protection

Training uses verified non-GAIA trajectories. The training collection rejects explicitly GAIA-sourced data and exact normalized GAIA question-hash overlaps. This does not claim to detect semantic paraphrase leakage. Gated GAIA question text and reference answers must stay in the evaluator environment rather than in published artifacts.

## Reading results

Use [docs/v3d-results.md](v3d-results.md) for the earlier controlled study and [docs/gaia-2026-11-23.md](gaia-2026-11-23.md) for the later matched GAIA100 result. The baseline and runtime revisions differ, so do not combine those two tables into one learning curve.
