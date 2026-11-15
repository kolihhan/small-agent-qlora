# PRO research gates — Qwen3.5-9B tool-agent QLoRA

[Canonical PRO method](https://github.com/kolihhan/kolihhan/blob/main/skills-search/pro-research-decision-gates.md): **P**roven failure → **R**eturn on experiment → **O**ut-of-sample proof. This is a concrete experiment decision record; it does not authorize new GPUs or overwrite training branches.

## Verified performance and failure mechanism

**Primary objective:** real autonomous closed-loop agent success and generalization to longer, unfamiliar tool workflows; GAIA is an external check. **Training is allowed only for the original Qwen/Qwen3.5-9B stack**, with training-source provenance protected from GAIA evaluation contamination.

| Evidence | Result | Decision |
| --- | --- | --- |
| [Frozen controlled200](https://github.com/kolihhan/small-agent-qlora/actions/runs/38752732871) | Base **69/200 = 34.5%** → v3B **105/200 = 52.5%** → v3D **124/200 = 62%** | Real but **synthetic task** gains; 17/19 v3D-over-v3B wins were `no_tool_stop`, not broad capability |
| [Matched GAIA100](https://github.com/kolihhan/small-agent-qlora/actions/runs/38911211243) | Base **9/100**, v3B **18/100**, v3D **18/100**; completion v3B **44%** → v3D **25%** | External v3D gain **zero**; do not promote v3D as GAIA champion |
| [Paired GAIA trace analysis](docs/gaia-v3d-results.md) | v3B final→v3D nonfinal **20** versus reverse **1**; v3D **876 tool calls** vs v3B **784**; failure recovery **0/25** in controlled test | **Over-search/termination + missing recovery**, not more steps or tool calls |
| v8 training-source/holdout audit | 304 parents / 798 labeled next actions, ~57% synthetic parents; mean **1.63 supervised tool actions** per train parent vs **4.03** in separate 35-parent teacher-forced holdout | Strong **horizon/exposure mismatch**; correlation with GAIA regression is hypothesis, not causal proof |
| [v8 T4 QLoRA action](https://github.com/kolihhan/small-agent-qlora/actions/runs/43671254880) | At last verified inspection the action was **in_progress**, **no adapter artifact** and **no 100-step proof** | **BLOCKED** until physical adapter and training report verified |

## Mandatory PRO gate for a new QLoRA candidate

1. **P:** Partition independent agent failures into `tool-selection/args`, `evidence-to-final`, `failure recovery`, `duplicate calls`, `over-search/termination`, `answer formatting`, `tool/environment failure` and `long-horizon context`. Use **closed-loop traces**, not only next-action labels. Explicitly ask whether the failure can plausibly be solved by QLoRA rather than a broken tool or evaluation harness.
2. **Training-data quality gate:** Every target action must have valid provenance, correct tool/evidence state and genuine next-action supervision. Include *source-diverse longer decisions*, actual failure/recovery and finalization examples before assuming short synthetic trajectories generalize. Disjoint **task families and protected evaluation questions**. Do not train on GAIA; do not mine GAIA100 mistakes into training templates.
3. **R: one change at a time.** Freeze train/heldout SHA, model exact ID, QLoRA rank/alpha/steps/context and seed. No additional synthetic expansion, SFT loop or GPU retry solely to improve supervised loss. Budget T4/HF/Actions usage and terminate repeated identical failure modes.
4. **Hardware evidence gate:** Successful workflow ≠ trained model. Require physical `adapter_model.safetensors`, nontrivial byte size, checked weight SHA, base-from-scratch attestation, **100 actual optimizer steps if claiming v8**, and coherent training-source SHA. If absent mark **BLOCKED**, never label “GAIA improved”.
5. **Diagnostic then closed loop:** Teacher-forced heldout next-action accuracy (including 35-parent v8 70 balanced checks) is **diagnostic**, not promotion. Compare exactly matched Base, v3B and v8 on **autonomous, source-/task-disjoint** non-GAIA multi-hop tasks, with longer 7–12-decision episodes and failure/recovery cases if independently available.
6. **Metrics:** Primary **end-to-end exact success** with paired rescues/regressions; also finalization/completion, duplicate blocks, wrong/no final, tool use/error counts, mean steps, cost/latency, and results by source and trajectory horizon. Report per-family counts and uncertainty; a one-task swing in 100 is not robust evidence.
7. **O / promotion:** Only if independent non-GAIA closed-loop transfer improves without serious stopping/recovery regressions, run a protected **matched GAIA external** comparison, preserving exact task IDs/runtime and **not tuning back to its exposed answers**. Historical GAIA v3B **18/100** is the external bar; do not substitute 62% synthetic or 70 teacher-forced actions. Demand actual overall improvement **and** no collapse relative to v3B's 44% completion/duplicate-call behavior; otherwise **REJECT**.
8. **Decision:** Existing v8 GPU in progress → **WAIT** and inspect final artifact before launching any competing GPU. If it fails, classify **allocation / OOM / kernel / checkpoint** from real logs and apply only a minimal, causally justified fix with controlled retry. If it completes but fails OOD, reject unmerged, do not tune GAIA-specific samples.

## Immediately actionable state

- Source of truth for v8 candidate and physical T4 action: [PR #34](https://github.com/kolihhan/small-agent-qlora/pull/34). Do not confuse passing unit tests with trained adapter.
- Original v3B still has the best matched **GAIA100 18/100 with 44% completion**; v3D is stronger on the controlled200 but is not better on GAIA.
- Next spend should be **real closed-loop capability measurement** after checkpoint, not another training stage to lower loss. Unknown GPU status must be rechecked, not assumed permanent.
