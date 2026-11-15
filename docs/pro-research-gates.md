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
| [v8 T4 QLoRA action](https://github.com/kolihhan/small-agent-qlora/actions/runs/43671254880) | **Reached 50/100 real optimizer steps** on T4; Colab session lost, no physical adapter weights exported | **BLOCKED_RESOURCE**, not OOM/finished v8; checkpoint absent |
| [v8 staged durable-35 retry](https://github.com/kolihhan/small-agent-qlora/actions/runs/43685165603) | Both actual T4 allocation attempts returned **Colab HTTP 503**, before optimizer-step 1; no adapter, no GAIA score | **BLOCKED_RESOURCE**; no more blind Colab reruns/paid GPUs without approval |
| [Real Qwen9B BASE autonomous MuSiQue8](https://github.com/kolihhan/small-agent-qlora/actions/runs/43689239130) | 3-hop **0/4**, 4-hop **0/4**; **0/8** correct, 7/8 final, 49 tool calls, zero execution errors or duplicate blocks | Real tool/answer failure, but **not** GAIA or Base-v8 paired proof; no specific evidence/answer failure mechanism shown by tool-name-only logs |

## Real closed-loop diagnostic follow-up (2026-11-15)

- The Base CPU experiment reached Qwen9B *actual* autonomous tool loops using physical SHA-verified, **20 original articles per MuSiQue task**, no teacher-forced prior actions. [PR #37](https://github.com/kolihhan/small-agent-qlora/pull/37). One model request stopped with `model_error`; seven produced answers, **none matched** their references. Tool names alone do **not** prove it read the supporting articles.
- **Physical v8 training-source audit:** immutable [source artifact 14124045608](https://github.com/kolihhan/small-agent-qlora/actions/runs/43619541894) contains **64 MuSiQue TRAIN parents** and **352 assistant `read` decisions in the recorded teacher traces**; exactly **128 `read` actions + 64 final answers** were selected for SFT (**192 labels**), while **160 other `read` decisions were deliberately excluded** because they were not sufficiently grounded by preceding visible evidence. This is a **next-evidence selection supervision gap**, not authorization to blindly label those 160 actions or to train on heldout/GAIA. First confirm read-path support coverage and independently validated action rationale; then propose a *new source-disjoint* supervised decision corpus.
- [PR #38](https://github.com/kolihhan/small-agent-qlora/pull/38) and [real action 43691026466](https://github.com/kolihhan/small-agent-qlora/actions/runs/43691026466) completed a **separate frozen 8-task** three/four-hop Qwen9B cohort, with no overlap: **0/8** strict exact, **7/8** finals, **38** real tool calls, **0** tool errors, **8/8** read the passage index, **6/8** had the expected reference literal somewhere in at least one visited article. The literal-in-read is a weak diagnostic only. **Three nonempty model answers contained the exact reference string** (*1909*, *Moscow*, *Yale Herald*) embedded inside longer explanations and thus failed strict answer-match; a fourth said *September 11 attacks* instead of *the 9/11 attacks*. Answer formatting is a demonstrable *partial* bottleneck, NOT overall semantic success.
- [PR #39](https://github.com/kolihhan/small-agent-qlora/pull/39) post-hoc [physical original-source action 43691933638](https://github.com/kolihhan/small-agent-qlora/actions/runs/43691933638) found **2/8** question cases visited every annotated supporting paragraph path; **6/8** missed >=1 annotated supporting passage, mean support-path recall upper bound **65.63%**. This supports an evidence-chain selection *hypothesis*. A selected path may deliver only the first 2000 chars and annotated supports are not a full causal proof; do not call this actual evidence recalled by the model. No model calls in audit.
- **Third cohort last8, FINAL**: [PR #40 closed/rejected](https://github.com/kolihhan/small-agent-qlora/pull/40), [physically completed paired run 43692339120](https://github.com/kolihhan/small-agent-qlora/actions/runs/43692339120) and [final artifact 14153223136](https://github.com/kolihhan/small-agent-qlora/actions/runs/43692339120/artifacts/14153223136) tested the remaining eight untouched 4×3-hop/4×4-hop MuSiQue TRAIN-derived tasks on identical Qwen9B Q4 model/workspaces with one opt-in generic short-final instruction. **Strict task success 0/8 baseline → 0/8 treatment; 0 rescues, 0 regressions**. Both arms finalized 8/8. Baseline answer included literal reference **7/8**, treatment only **4/8** (not a semantic metric). Tool calls 59→53, latency sum 2690.0→1988.1s; these do not establish stable speed or task success. Fixed gate failed; **no production merge, no more prompt tuning on any of the now-exposed 24 MuSiQue holdout tasks**. This is not GAIA nor matched v8 adapter evaluation.
- **Do not tune prompts, replay exposed GAIA answers, or generate imitation SFT from these 8/16 labels.** If support paths are not visited, research transferable evidence selection; if all support paths are visited yet answers fail, investigate actual delivered context and multi-hop answer composition before allocating GPU.
- Native TRL `DataCollatorForLanguageModeling` uses supplied tokenized `labels` when present, and `SFTTrainer` skips re-tokenizing datasets already containing `input_ids`; the existing training implementation constructs prompt-masked labels. Do not assume the training objective ignores masks without a pinned on-device batch proof. This is **not** a validated agent gain.

## Mandatory PRO gate for a new QLoRA candidate

1. **P:** Partition independent agent failures into `tool-selection/args`, `evidence-to-final`, `failure recovery`, `duplicate calls`, `over-search/termination`, `answer formatting`, `tool/environment failure` and `long-horizon context`. Use **closed-loop traces**, not only next-action labels. Explicitly ask whether the failure can plausibly be solved by QLoRA rather than a broken tool or evaluation harness.
2. **Training-data quality gate:** Every target action must have valid provenance, correct tool/evidence state and genuine next-action supervision. Include *source-diverse longer decisions*, actual failure/recovery and finalization examples before assuming short synthetic trajectories generalize. Disjoint **task families and protected evaluation questions**. Do not train on GAIA; do not mine GAIA100 mistakes into training templates.
3. **R: one change at a time.** Freeze train/heldout SHA, model exact ID, QLoRA rank/alpha/steps/context and seed. No additional synthetic expansion, SFT loop or GPU retry solely to improve supervised loss. Budget T4/HF/Actions usage and terminate repeated identical failure modes.
4. **Hardware evidence gate:** Successful workflow ≠ trained model. Require physical `adapter_model.safetensors`, nontrivial byte size, checked weight SHA, base-from-scratch attestation, **100 actual optimizer steps if claiming v8**, and coherent training-source SHA. If absent mark **BLOCKED**, never label “GAIA improved”.
5. **Diagnostic then closed loop:** Teacher-forced heldout next-action accuracy (including 35-parent v8 70 balanced checks) is **diagnostic**, not promotion. Compare exactly matched Base, v3B and v8 on **autonomous, source-/task-disjoint** non-GAIA multi-hop tasks, with longer 7–12-decision episodes and failure/recovery cases if independently available.
6. **Metrics:** Primary **end-to-end exact success** with paired rescues/regressions; also finalization/completion, duplicate blocks, wrong/no final, tool use/error counts, mean steps, cost/latency, and results by source and trajectory horizon. Report per-family counts and uncertainty; a one-task swing in 100 is not robust evidence.
7. **O / promotion:** Only if independent non-GAIA closed-loop transfer improves without serious stopping/recovery regressions, run a protected **matched GAIA external** comparison, preserving exact task IDs/runtime and **not tuning back to its exposed answers**. Historical GAIA v3B **18/100** is the external bar; do not substitute 62% synthetic or 70 teacher-forced actions. Demand actual overall improvement **and** no collapse relative to v3B's 44% completion/duplicate-call behavior; otherwise **REJECT**.
8. **Decision:** v8 GPU run **failed after 50 steps** with a lost Colab session; both attempts to allocate T4 for the early step-35 durable checkpoint failed **HTTP 503 before training**. Actual model training checkpoint remains **absent**. No repeated blind Colab retry, no paid GPU without authorization, no GAIA claim until a full physically verified adapter and matched OOD comparisons. If GPU allocation becomes available, resume only with SHA-verified adapter+optimizer/scheduler/RNG and an unchanged training goal.

## Immediately actionable state

- Source of truth for v8 candidate and physical T4 action: [PR #34](https://github.com/kolihhan/small-agent-qlora/pull/34). Do not confuse passing unit tests with trained adapter.
- Original v3B still has the best matched **GAIA100 18/100 with 44% completion**; v3D is stronger on the controlled200 but is not better on GAIA.
- Physically verified **v8 checkpoint absent**; [PR #36](https://github.com/kolihhan/small-agent-qlora/pull/36) checkpoint export code exists, but T4 allocation twice failed. Unpaid GitHub CPU **Base-only real MuSiQue8 score 0/8**, not GAIA. Diagnostic read-path and source-support studies in PR #38/#39 are nonpromotional; wait for their actual artifacts before inferring cause. Avoid blindly creating more synthetic training trajectories.

## Source hyperlink positions in independent IIRC evaluation

The LLM Prompt Recovery fifth-place team treated evidence retrieval, candidate
generation and scoring as distinct failure points and checked the T5
implementation after a local gain failed to transfer. The Konwinski Prize host
froze submissions before collecting final GitHub issues. Our application is to
alter only a source-native evidence signal and measure delivered evidence
separately from exact task completion. Sources:
[Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery/writeups/z-d-z-5th-place-solution)
and [Konwinski Prize data protocol](https://www.kaggle.com/competitions/konwinski-prize/data).

**P.** The PR #40 answer-only task trial scored 0/8 in both arms. PR #39
established that the model omitted at least one annotated passage path on six
of eight different MuSiQue episodes. A path alone cannot establish that the
right text reached the model. Two hypotheses remain: the agent cannot connect
a clue to the next source, or it gets the source text and fails to combine it
or stop with the correct answer.

**R.** On eight frozen [IIRC](https://aclanthology.org/2020.emnlp-main.86/)
DEV questions from distinct primary pages, compare a flat directory against the
same full local documents with original hyperlink destinations annotated inline
in the primary page. Include all usable original link positions without
consulting question-level labels at inference. Both arms run the original
Qwen3.5-9B manifest and AgentRuntime with one local read tool, eight turns,
the same user input and strict answer scoring. Freeze archive checksums,
all workspace bytes and the eligible two-to-three-linked-source slice first.
Report exact success and paired flips with model-visible annotated spans
and completion/tool use. The preregistered positive screen requires at least
two rescues, zero regressions, at least two more episodes receiving all
annotated evidence and no completion decline.

**O.** IIRC is a different published benchmark from MuSiQue. The selected
primary and supporting titles exclude IIRC TRAIN primary titles. The evaluation
conditions on evidence within an initial tool observation to control for
pagination; it does not estimate whole-IIRC accuracy or GAIA transfer. This
local read task is narrower than the full GAIA tool surface.
A positive result can motivate a separate TRAIN-only label feasibility audit
and longer real closed-loop evidence, but cannot qualify a QLoRA adapter.

**Result — REJECT inline source hyperlinks as the standalone intervention.**
All four paired shards completed on CPU. Strict success was **0/8 → 0/8**,
with zero rescues and zero regressions. Both arms recorded 8/8 runtime finals.
Actual tool observations contained **27/29 annotated spans in both arms**.
Full annotated text arrived in **6/8 → 7/8** episodes, below the two-task
evidence gate. Calls were **34 → 33**, and summed runtime was 1053.5s → 1042.9s.
[Run 43700029046](https://github.com/kolihhan/small-agent-qlora/actions/runs/43700029046) ·
[Raw paired artifact](https://github.com/kolihhan/small-agent-qlora/actions/runs/43700029046/artifacts/14156237462).

**Failure attribution.** PR #40 was independently rechecked against the exact
original MuSiQue paragraph bytes and published support labels. The original
arm visited all annotated paths in **4/8** episodes (21/28 paths), versus
**2/8** (20/28) with the short-final prompt. The old trace stores paths, so
these counts upper-bound prior evidence delivery. The IIRC trace measures
actually delivered annotated text. Examples `q_12134`, `q_11664`, and
`q_11268` support the correct comparisons in both arms, with the reference
entity expressed inside a longer answer. `q_10893` illustrates ambiguity
between the founding of an institution and the opening of its current building.
`q_11543` asks for a year and the public reference is a seminary name. Keep
the full cohort and original strict scorer. These observations support a
mixture of evidence access, answer selection, response format and reference
quality. The IIRC cohort favors evidence that fits in an initial observation.
The standard read tool also retains URL capability; only matching local
observations were counted and external attempts were not separately preserved.

**Training-data feasibility.** Replayed actual bounded ReadTool calls on
128 hash-selected IIRC TRAIN primary pages (357 candidate questions). The
audit excluded protected IIRC DEV identities and known overlapping titles
from the frozen DEV workspaces and the previous MuSiQue24 workspaces. Seven
parents supplied 14 physically verified linked read targets and seven
evidence-complete final states. Each eligible next-read target has a published
anchor in visible source text and a matching support span returned by its
tool read. The audit allows any verified unvisited target; the source does
not prescribe a unique reading order. Allowing up to three pages by following
the actual emitted `next_offset_chars` reduced missing-support exclusions
from 26 to 15, with still **7 eligible parents**, all short. There were
**zero** eligible 7–8-decision parents. This is a feasibility screen with zero
training labels exported. Keep the 160 ungrounded MuSiQue reads excluded.

**Training decision — BLOCKED.** The independent source did not yield
enough authentic long trajectories. Use matched under-evidenced and
evidence-complete states to test continuation versus finalization before
training, with genuine tool observations and separate source families.
A real candidate requires persistent adapter, optimizer, scheduler and RNG
state. Rechecked both historical T4 runs. The original run artifact contains
only a runtime notebook, and the Stage35 retry produced no checkpoint artifact.
The matched GAIA100 bar remains **18/100 with 44% completion** for v3B.
