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

## Matched complete and incomplete evidence states

The IIRC closed-loop study observed correct conclusions in verbose finals and
incomplete or wrong conclusions even after source delivery. This motivates
separating a decision to continue reading from a decision to submit a
task-shaped answer. Use frozen public [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG)
news sources, independent of the previous Wikipedia evaluation. Verify the
provided support spans in original source articles. Freeze four 3-document
and four 4-document cases with disjoint supporting articles and a decisive span whose
answer literal is absent from the question and other visible passages.
The task slice conditions on this literal distinction.

Give each case identical candidate article paths, publishers, and dates, with
headlines omitted to keep the answer out of the inventory. Use two physically replayed
starting observations with the same number of files. In the incomplete arm,
substitute one same-category distracting passage for the decisive passage.
Compare the original read or free-text AgentRuntime with a Qwen grammar-
constrained read or submit action. Both have the same source files, Qwen
manifest, three subsequent model turns and token limit. Read only frozen
local files. Measure first useful reads, premature submissions, exact
task completion supported by all required files, ordinary exact match,
tool calls, and elapsed time. Use the same scorer. The typed-arm screen
requires at least three net evidence-supported exact gains, at most one
regression, rescues in both states, no completion regression in either
state, and no worse first useful reads or premature submits when incomplete.
These are paired seeded recoveries. A positive result merits cold-start
confirmation and source-shift tests before agent integration or QLoRA.

**Result — REJECT the joint typed read/submit policy.** With complete
evidence, native and typed evidence-supported exact success were **0/8 →
6/8**. With missing evidence, supported success stayed **0/8 → 0/8**.
The native policy read the withheld source first in **7/8 → 0/8** across
arms. Premature submissions were **1/8 → 8/8**. The typed arm made zero
additional reads in either state. Both arms finished all episodes.
All six typed exact answers in the complete state also scored exact on
the same six tasks without the decisive submitted evidence. The task
identities and strict outcomes are preserved. This rules out attributing
the exact-score gain to successfully identifying missing evidence. It
does not distinguish prior knowledge from clues in the remaining text.
The run recorded scores and read/stop outcomes and did not store final
answer text. Both states begin from controlled reads, rather than cold
search. [CPU run](https://github.com/kolihhan/small-agent-qlora/actions/runs/43703160088) ·
[Final artifact](https://github.com/kolihhan/small-agent-qlora/actions/runs/43703160088/artifacts/14158056217).

Keep the original native tool policy as the control. A later test can
bind the final rendering boundary after a native stop and independently
check groundedness with pairs where changing delivered evidence changes
the correct answer. The current result does not establish a long-horizon
training signal or an adapter result. New QLoRA remains blocked on verified
decision supervision and checkpoint evidence; the matched GAIA100 bar is
18/100.

## New independent TRAIN-source readiness audit — 2Wiki (2026-11-15)

A physical CPU-only [source-feasibility run 43707596286](https://github.com/kolihhan/small-agent-qlora/actions/runs/43707596286) used **2WikiMultiHopQA TRAIN** at Hugging Face source revision `fe713bfbd1afbca1a65246741a75890405d56a3a`. It checked the original source-order first **1,024** tasks (not a random population estimate), publisher supporting sentence indices, and the actual 2,000-character observation bound. All 1,024 source identities/annotations passed structure validation; focused tests **4 passed**. Genuine annotated supporting-document counts were **819 with two, 205 with four**. Literal-title reachability plus bounded support delivery completed **897/1,024**; individual reachable/delivered supporting-doc counts were **2,338/2,329**. Critically, **0** cases had six distinct delivered support documents, so the sample cannot supply six distinct document reads + final as a seven-decision supervised trajectory.

**Decision: SOURCE_INSUFFICIENT for our long-horizon training-source requirement.** [PR #43](https://github.com/kolihhan/small-agent-qlora/pull/43) retains code and tests without adopting an agent policy. **Zero labels exported, no adapter trained, no GAIA improvement**. Source supporting-fact annotations are not next-action oracle labels; a document title being literal-reachable or its span delivered does not prove unique beneficial next read. Do not manufacture 7–8 steps by adding unneeded rereads or promote this short-path slice into v8 training. This does not reject all 2Wiki TRAIN or potential shorter-span composition tasks. The stronger next test needs an independently sourced real next-action decision with different correct actions under matched *delivered* incomplete versus complete evidence, then genuine closed-loop transfer before any GPU attempt.


## Source-justified 2Wiki search/read path count vs actual agent skill (2026-11-15)

A second source-pinned [CPU audit #43709834019](https://github.com/kolihhan/small-agent-qlora/actions/runs/43709834019) revisited the same first 1,024 original 2WikiMultiHopQA TRAIN rows at revision \`fe713bfbd1afbca1a65246741a75890405d56a3a\` using *non-oracle title retrieval and source read* pairs. The local index contains publisher-provided distractors as well as supporting pages; the title chosen for every search is anchored to the question or a previous actually delivered read. Published supporting-fact labels only enter post-hoc accounting. Four unit tests passed.

**896/1,024** paths delivered all annotated supporting evidence in the bounded read observations; of 205 four-support-document tasks, **156** were evidence complete; **87** original TRAIN rows were eligible for at least eight nonredundant search/read steps under a source-text-only policy. The result corrects the earlier mistaken assumption that a seven-action task needs six **different documents**. Search and then read can be separate, meaningful actions when the filename is not supplied upfront. However the local full-text title index is a simulated source environment, not an observed public-web retrieval tool and not autonomous Qwen decisions.

**TRAINING STILL BLOCKED.** These are feasible action paths, not a validated teacher dataset or a trained policy. No action labels exported and no adapter. To determine what QLoRA should learn, the original Qwen model must attempt independent source tasks with native search/read, trace exact delivered evidence and stop reasons, and demonstrate actionable next-state errors. A separate 8-task, published **2Wiki VALIDATION** source-disjoint autonomous baseline has been registered under [PR #44](https://github.com/kolihhan/small-agent-qlora/pull/44); it does not score GAIA or support an official gain until its physical Actions run completes.


## Physically verified source-disjoint 2Wiki VALIDATION native Qwen baseline (2026-11-15)

The first independent **8 original 2WikiMultiHopQA VALIDATION** four-support-document tasks (8 distinct IDs and source support titles) were frozen before any model decision. [Original actual free-CPU Qwen9B run #43710360733](https://github.com/kolihhan/small-agent-qlora/actions/runs/43710360733) fully completed four inference shards and aggregate with the unchanged native AgentRuntime, source-local title search and bounded ReadTool (NOT public-web search). Seven focused unit tests passed; same original Qwen9B pinned digest as historical evaluations. Results: **strict exact 3/8**, **finalized 8/8**, **delivered all four annotated supporting source documents 0/8**, **finished without complete published support 8/8**, **31 tool calls**, **0 tool errors**. [Raw answers/paths artifact](https://github.com/kolihhan/small-agent-qlora/actions/runs/43710360733/artifacts/14160611243); final [PR #44](https://github.com/kolihhan/small-agent-qlora/pull/44), closed unmerged after evidence was recorded.

Per-case traces show the native Qwen agent usually searches for two entities named in the user question, reads the corresponding movie pages, then finalizes without reading the additional director biography facts in the publisher's support set. A few answers have the published option within an explanatory sentence but are not strict exact; literal-string presence is **NOT** final answer correctness. Conversely, lack of all four annotated supports alone does not prove *every* final was epistemically unjustified: minimal sufficient evidence could be shorter than gold support. The finding is a failure mechanism candidate, **NOT a trained adapter gain**, GAIA score or an identified unique good training action.

A second **non-overlapping** eight-case VALIDATION test [PR #45](https://github.com/kolihhan/small-agent-qlora/pull/45) was preregistered to compare unchanged original Qwen AgentRuntime against a first-party article-title-path metadata affordance based only on titles literally present in previously **visible** original read text, no task answer or gold support in the tool. Screen requires 2 net strict exact rescues with 0 regressions **AND** 2 more full source-support cases, without increasing missing-support finishes. Do not tune the first now-exposed eight-case cohort; no model training until a genuine source-disjoint native success mechanism is shown.


## Source-mentioned read links: evidence improves, strict end-task gate does not (2026-11-15)

A second original **2WikiMultiHopQA VALIDATION** 8-task four-support cohort, hash-selected and with all eight previously evaluated identities and all **32** prior supporting physical paths excluded, compared unchanged Qwen3.5-9B native AgentRuntime/Search/Read to the same system with only first-party link metadata at a real \`read\` boundary. Related article paths were offered only if that local article title was literally present within the first 1,200 characters of the current actually read document; it was NOT a gold-support-label hint or forced read.

Four genuine Qwen CPU paired shards were all successful in [Actions #43712873154](https://github.com/kolihhan/small-agent-qlora/actions/runs/43712873154). A wildcard aggregate initially included the frozen cohort JSON as a fifth input and failed; **no original Qwen episode failed or was rerun**. The exact four source-pinned original model outputs were recovered and aggregated in the successful [CPU-only aggregate #43714195334](https://github.com/kolihhan/small-agent-qlora/actions/runs/43714195334) with [complete original per-case artifact #14161871245](https://github.com/kolihhan/small-agent-qlora/actions/runs/43714195334/artifacts/14161871245). Eight of eight episodes scored in both arms.

Measured native vs linked: strict exact **2/8→3/8**, **1 rescue / 0 regressions**; full delivery of all four source-publisher annotated support documents **0/8→5/8**; completed **8/8 in both**; Qwen tool calls **34→46**, tool errors 0→0; total runtime **784.6→1125.8 seconds**. The preregistered out-of-sample decision gate required **>=2 NET exact rescues** and +2 full-support cases, no regressions and no completion loss. It **FAILED** on only +1 strict exact, despite real 5-case evidence-delivery gains. **Do not promote the link tool policy or call it a new QLoRA/GAIA result.** [PR #45](https://github.com/kolihhan/small-agent-qlora/pull/45) closed unmerged.

A distinct post-hoc diagnostic: three of the five linked-arm *evidence-complete* cases had a correct final film explicitly named at the **end** (two \`Answer:\` lines and one standalone final line) but failed the original strict exact scorer because the Agent returned explanations as well. This does **not** authorize arbitrary substrings or gold-aware answer parsing: a referenced choice can appear in a WRONG explanatory paragraph. It identifies **terminal response shape** as a separate bottleneck alongside evidence selection. Before adopting a generic final-only rendering layer, freeze a source-independent extraction contract and validate on a *new* unseen data/source family with false-positive controls and no regressions. Do not refit source-link behavior or answer formatting to these now-exposed 8 film-director-comparison questions.

Agent learning objective is still actual closed-loop success under novel tools and evidence, not article-count proxies, teacher-forced next-actions, answer-string presence, or synthetic steps. GPU QLoRA remains blocked on verified long source-justified action trajectories and independent transfer; matched v3B GAIA100 remains **18/100**.


## Generic final-output extraction — separate bottleneck, fails source-family screen (2026-11-15)

The original source-anchored local-link Qwen9B VAL8 study [PR #45](https://github.com/kolihhan/small-agent-qlora/pull/45) showed real source retrieval gains **0/8→5/8** complete support and **2/8→3/8** strict task exact, but the preregistered **2-rescue** end-goal gate failed. Three of the five new evidence-complete outputs were semantically correct at the explicitly marked final answer line but verbose, so they failed original strict exact. This was a distinct post-hoc observation; it does not imply source competence or justify tuning that exposed test cohort.

A source-agnostic, deterministic *final answer field* extractor was then frozen **before opening any IIRC or MuSiQue output artifacts**. It accepts only an explicit \`Answer:\` final field or a blank-line-delimited short standalone last line; never scans the answer for an arbitrary reference substring, never reads gold/question/tool trace, and never calls another model. Eleven generic unit tests passed. No \`AgentRuntime\` default changed.

**Original independent source-family physical offline replays**:
- [IIRC original completed Qwen8 source, run #43714884751](https://github.com/kolihhan/small-agent-qlora/actions/runs/43714884751): flat **0/8→1/8**, original linked **0/8→1/8**, both on the **same unique source case** \`q_11727\`; zero regressions, no model calls. [Full per-case artifact #14162545380](https://github.com/kolihhan/small-agent-qlora/actions/runs/43714884751/artifacts/14162545380).
- [MuSiQue original Qwen other8, run #43715087286](https://github.com/kolihhan/small-agent-qlora/actions/runs/43715087286): **0/8→0/8** strict; 4 actual answer strings changed, **0 rescues/0 regressions**, no model calls. [Full per-case artifact #14161823718](https://github.com/kolihhan/small-agent-qlora/actions/runs/43715087286/artifacts/14161823718).

**Decision — REJECT generic renderer promotion, [PR #46](https://github.com/kolihhan/small-agent-qlora/pull/46) closed unmerged.** The prereg IIRC screen required >=2 rescued *distinct* source questions; only one unique rescue occurred. MuSiQue cross-family produced none. This is a narrow formatting intervention, not actual QLoRA transfer or autonomous improvement, even though it illustrates a separately measurable finalization bottleneck. Do not revisit exposed IIRC/MuSiQue cases to tune the renderer. The remaining research bottleneck is verified long-horizon decision supervision and source-independent evidence-based answer selection/recovery. **No new QLoRA adapter was trained, no GAIA100 matched score beyond v3B 18/100; no SQL training.**


## Original real-tool 2Wiki source audit: zero SFT-ready long demonstrations (2026-11-15)

One conservative candidate-generation hypothesis was a four-support-document source-justified navigation path with four real nonredundant \`search\` + \`read\` pairs and a decisive final. [Physical CPU audit #43717931437](https://github.com/kolihhan/small-agent-qlora/actions/runs/43717931437) ran on first **5,000 publisher-order original 2Wiki TRAIN** examples, pinned \`fe713bfbd1afbca1a65246741a75890405d56a3a\` and the same 2,000-char actual tool observation bound. Four focused tests passed. **511/5,000** were potential eight-action navigation candidates by purely lexical, gold-blind lookahead; **0/511** passed real bounded \`LocalSearch.run\` + \`ReadTool.run\` and post-hoc publisher-support verification. Rejections: 192 redundant reads (search preview already supplied their cited span), 84 accidental distractor chosen, 212 reference exposed in question/invalid, 18 answer already visible before final supporting read, 5 target not delivered; 4,489 other rows failed long-source preflight. No supervised training example exported; zero new Qwen decisions or GPU training steps. The [artifact](https://github.com/kolihhan/small-agent-qlora/actions/runs/43717931437/artifacts/14163281589) holds summary and empty JSONL (SHA256 \`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\`). [PR #47](https://github.com/kolihhan/small-agent-qlora/pull/47) closed unmerged.

**Key correction:** a search followed by read counts as two action timestamps, but not necessarily two *informationally justified* learning decisions. The previous 87 feasible 2Wiki navigation paths overstated credible useful-action supervision when the search result already contained the publisher cited sentence. This strict pipeline rejects this source route as a stand-alone long-horizon SFT corpus; it does not prove all 2Wiki data or all useful read actions are impossible. Scripted navigation with post-hoc source support is NOT an expert agent demonstration; no model was retrained and matched v3B GAIA100 remains 18/100.

Next source lane: inspect real published, executed successful tool-use traces from another family, with original adjacent tool-call/observation pairs and publisher task outcomes. Keep success labels marked *publisher reported, not independently replayed*, and do not transplant OpenHands tool-call schemas directly into GAIA native tool policy. This is source research, not permission to spend GPU credits or train an adapter.


## First actual long, source-executed web-agent SFT evidence (2026-11-15)

**2Wiki script-path correction:** original first 5,000 2Wiki TRAIN source rows gave 511 candidate nonoracle search/read paths but **zero** source-verified eight-useful-action trajectories once actual search snippets, tool responses and nonredundancy were checked. Earlier path length was not a credible teacher reward. Do not resubmit dummy reads or treat raw source paragraph support as next-tool expert labels. See [run #43717931437](https://github.com/kolihhan/small-agent-qlora/actions/runs/43717931437) and closed [PR #47](https://github.com/kolihhan/small-agent-qlora/pull/47).

**Original actual OpenHands executed trajectories are real, but differ from GAIA tools.** [Nebius physical pinned-source audit #43720847680](https://github.com/kolihhan/small-agent-qlora/actions/runs/43720847680) used first 500 original SWE-rebench Qwen3-Coder-480B OpenHands episodes at source revision \`35455389ab51bf5e2306bfd436ef72d0f98bf882\`. The original OpenHands **finish** action has no following tool result; previous 0-coverage parser erroneously classified this valid terminal call as a dropped observation. After correcting and unit-testing it: **445/500** fully paired executed normal tool outputs, **231/500** publisher-labeled successful episodes with >=7 real normal tool calls, >=3 assistant action turns, full feedback and a nonempty patch. The 231 contain **12,965** real call→feedback pairs over **224 unique source issues** and **177 repos**; the original success label is **not independently reproduced**. Original tools \`execute_bash\`, \`str_replace_editor\`, \`think\`, \`task_tracker\` are **not** the source-compatible web research tools, much less the unchanged GAIA runtime. [PR #48](https://github.com/kolihhan/small-agent-qlora/pull/48), metadata-only public artifact. Do **not** teach native GAIA that OpenHands coding tools exist without an actual harness and independent eval.

**Original Apache web-agent trajectories have useful real source/search signal.** Original [NextSearch-1 Trajectories](https://huggingface.co/datasets/NextTokenAI/NextSearch-1-Trajectories) pinned revision \`d32183c6289d32c5d51b6247e68d53a58dddd495\`, **Apache-2.0 \`sft-trajectories\` TRAIN only**, all **3,270** episodes. [Physical source audit #43721683485](https://github.com/kolihhan/small-agent-qlora/actions/runs/43721683485): all 3,270 original \`search/fetch\` actions have their actual nonempty tool messages; **1,033** final answers match publisher reference strictly; **462** have at least seven actual tool calls, and **141** meet seven-call, real-search, real-fetch, strict-answer + three-turn gates (**138 unique publisher task IDs**, 321 source rows duplicate task IDs). **328** have no source reference and are *unscorable*, not failures or successes. Original source \`search(objective, search_queries)\`, \`fetch(objective, urls, offset)\` does **not** match native \`search(query)\` or \`read(source)\`: do **not** feed raw action targets directly into current GAIA production QLoRA. [PR #49](https://github.com/kolihhan/small-agent-qlora/pull/49) includes first-party schema tests and a separate experiment-only actual-web adapter that delegates these calls to the existing SearchTool and ReadTool rather than inventing tool output or modifying the original GAIA agent.

**The first source-verified SFT candidate file exists, but is not a trained adapter.** [Physical source curation #43723143610](https://github.com/kolihhan/small-agent-qlora/actions/runs/43723143610) scanned all 3,270 original Apache TRAIN records and emitted **110 distinct task parents, 1,086 original source tool calls, 452 selected actual assistant supervised turns**. Original source reasoning_content excluded; observations physically limited to native 2,000 characters; parent rejected when source reference could not be seen in those bounded original tool replies, the answer was already in the question, source tool had errors, or task ID repeated. Candidate JSONL SHA256 \`abb15668cd5c6dbf039872686d9178e052b0e4b26e66be65823546b00dca66dd\`. Focused test validates that the existing \`load_verified_rows\` provenance contract accepts the source-level verified row. \`verified=True\` in these rows means **original publisher reference match plus original tool responses**, NOT independently reenacted answer accuracy, NOT GAIA correctness, NOT deployed skill. [Candidate artifact #14165800164](https://github.com/kolihhan/small-agent-qlora/actions/runs/43723143610/artifacts/14165800164).

**Actual independent native model check is running, no outcomes claimed yet.** Eight **NextSearch VALIDATION** task IDs were hash-frozen, excluding all original TRAIN identities, and spanning ASearcher and TaskCraft. A new experiment-only source-compatible \`search/fetch\` tool adapter passed 12 focused tests. Unmodified Qwen3.5-9B \`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90\` is executing real public-web searches/fetches on four free-GitHub-CPU shards under [run #43722732105](https://github.com/kolihhan/small-agent-qlora/actions/runs/43722732105), source answers only used after final output. The first completed shard was 0/2 correct with no tool errors. Do not claim success until all eight scored.

**GPU QLoRA is still gated on actual tokenizer feasibility and independent end-task outcomes.** A CPU source-only [actual Qwen3.5-9B token-context preflight](https://github.com/kolihhan/small-agent-qlora/actions/runs/43723477513) is validating the native Qwen chat template and intact prompt/completion token lengths for all 110 parents. The first attempt failed only because Jinja2 was omitted from the tokenizer-only GitHub environment, not due to a data contract violation; exact dependency was added and rerun. Do not train with silent prompt truncation, zero model improvements, or guessable adapter checkpoints. Historical v3B matched GAIA100 remains **18/100** until a successful adapter and source-/GAIA-disjoint paired end-to-end evaluation demonstrate a difference.


## Real web-agent source states: exact Qwen9B VAL baseline and 4K training pack (2026-11-15)

**Data, not model gains:** Original pinned Apache-2.0 NextSearch TRAIN source \`d32183c6289d32c5d51b6247e68d53a58dddd495\` yielded 110 original full tool-action parents with publisher-reference correctness and actually observed \`search/fetch\` messages [source curation #43723143610](https://github.com/kolihhan/small-agent-qlora/actions/runs/43723143610). Original 2,000-char bounded tool observations and real publisher final answers were retained; hidden reasoning_content excluded.

**Independent live Qwen9B baseline is real and negative.** Eight original NextSearch **VALIDATION** task IDs excluded all original TRAIN IDs and used a separate source-compatible, real-web \`search(objective,search_queries)\` + \`fetch(objective,urls,offset)\` adapter that leaves the production GAIA \`search(query)\` + \`read(source)\` tool implementation unchanged. [Four actual Qwen9B CPU shards and successful aggregate #43722732105](https://github.com/kolihhan/small-agent-qlora/actions/runs/43722732105), digest \`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90\`, **0/8 strict exact**, **8/8 final completed**, **43 real tool calls (31 search, 12 fetch), 0 tool errors**, 2,233.3 seconds summed runtime. Sources: six ASearcher, two TaskCraft. This is **NOT GAIA**. There is no QLoRA adapter improvement claim.

**Genuinely different evidence states, with no invented action labels.** [True-source causal window construction #43724022656](https://github.com/kolihhan/small-agent-qlora/actions/runs/43724022656) exported **173** actual source-next-action windows (79 evidence missing → original next tool action; 94 source answer visible → original final), across 110 original parents; **74 parents have both states**. Another 16 parents were flagged as possible early over-search after source answer was already seen and excluded from terminal-final teacher exposure; tool observations remained real source output. These are **partial historical context windows**, not an oracle expert policy.

**Required full-prompt Qwen token safety.** [Native Qwen3.5-9B tokenizer on exact 173 windows #43725714707](https://github.com/kolihhan/small-agent-qlora/actions/runs/43725714707) demonstrated full-context example counts **0/173 at 1,024**, **0/173 at 2,048**, **62/173 at 4,096** (28 next-tool, 34 final), and **172/173 at 8,192** (78 next-tool, 94 final). Existing v3B's 1,024-token setting cannot legitimately train ANY of these source-state windows without removing real evidence. Never silently truncate or suggest v3B's existing adapter was trained on this source.

**Actual balanced 4K training-input artifact (NO TRAINING):** [run #43726123262](https://github.com/kolihhan/small-agent-qlora/actions/runs/43726123262) passed 8 source safety tests and selected **48 complete-source-window supervised next actions** (24 continue tool, 24 actual final) from **28 unique original source questions**, **20 paired state families**, actual Qwen3.5-9B tokenizer revision \`c202236235762e1c871ad0ccb60c8ee5ba337b9a\`. Total source SHA256 \`f15b8253507aa21e56f96627a7d816a2835e356237302a103380bca64f3dfc2b\`; selected 48-row JSONL SHA256 \`5420cfdb6111806a23b4bf6cb9cf04c77c75b5f0cf4af6c0b63727f312f40432\`; [candidate artifact #14166139977](https://github.com/kolihhan/small-agent-qlora/actions/runs/43726123262/artifacts/14166139977). Hash-ordered selection is blind to VAL gold/result. **All 48 have preserved, actually observed source context and assistant completions; none are GAIA outcomes.** A tiny experimental Qwen9B 8-step QLoRA compatibility program, pinned source SHA and unchanged source tools, is prepared on the branch only; no trained adapter exists.

**Actual GPU blocker (do not misreport as training):** [free HF quota check #43726189255](https://github.com/kolihhan/small-agent-qlora/actions/runs/43726189255) confirmed GitHub HF_TOKEN belongs to \`kolihhan\` and 300s free ZeroGPU quota; but attempted creation of **new isolated private Gradio ZeroGPU Space** [#43726546183](https://github.com/kolihhan/small-agent-qlora/actions/runs/43726546183) failed at the HF create repository API with explicit **HTTP 402 Payment Required / requires PRO for new Gradio/Docker Spaces**. **No GPU job, optimizer step, billable hardware, or new Space** was launched. Existing account contains two PRIVATE legacy Spaces \`kolihhan/small-agent-qlora-v2-train\` (contains prior adapter and frozen train() API) and \`kolihhan/small-agent-qlora-v7-batch-trainer\` (v7-specific API). [Read-only inventory #43726713362](https://github.com/kolihhan/small-agent-qlora/actions/runs/43726713362) and [function/file inspection #43726833715](https://github.com/kolihhan/small-agent-qlora/actions/runs/43726833715) verified neither exposes a generic upload/train endpoint. **Do not overwrite existing Spaces or adapters** to bypass HF policy, and do not request paid HF GPU Jobs without approval. Use an authorized existing Colab GPU runtime or a newly available free resource to execute the pinned 48-row pack at 4,096 tokens, with short 8-step checkpointed pilot, then re-evaluate independent VAL and non-GAIA OOD (plus unchanged GAIA control) before promotion.

**Status:** Last matched GAIA100 QLoRA v3B **18/100**, original BIRD100 Text-to-SQL **39/100** unchanged; no SQL model training. Do not call this a verified learning breakthrough.

## Colab-only Qwen3.5-9B 4K QLoRA next step; original results still unchanged (2026-11-15)

The user chose **Google Colab** rather than Hugging Face ZeroGPU or paid HF Jobs. The source artifact and training controls remain frozen: 48 original bounded NextSearch action windows, 24 missing-evidence tool decisions + 24 evidence-present final decisions, 28 independent source parents (20 paired), original TRAIN only; exact source JSONL SHA256 **5420cfdb6111806a23b4bf6cb9cf04c77c75b5f0cf4af6c0b63727f312f40432**, original Qwen3.5-9B model revision **c202236235762e1c871ad0ccb60c8ee5ba337b9a**, max 4096 tokens with full prompt mandatory. Free HF new Space attempt produced HTTP 402 and no GPU training; do not modify existing v3B/v7 Space or use paid jobs without approval.

A ready Colab notebook is committed on the exploratory [PR #49](https://github.com/kolihhan/small-agent-qlora/pull/49) branch: [NextSearch_Real_State_Qwen9B_QLoRA_v0_Colab.ipynb](https://github.com/kolihhan/small-agent-qlora/blob/experiment/nextsearch-gold-grounded-search-traces-20261115/notebooks/NextSearch_Real_State_Qwen9B_QLoRA_v0_Colab.ipynb). It retrieves the exact GitHub source artifact (read-only PAT from the user's own Colab Secret), checks SHA and 24:24 source-state labels, pins model revision, checks CUDA/VRAM, and trains a short 8-step 4-bit LoRA pilot only on user-selected GPU; every 4 steps checkpointed to a dedicated Google Drive folder, with resumable checkpoints and explicit final output checks. L4/A100 tries full all-linear LoRA. A possible T4 route uses smaller q_proj/v_proj LoRA modules but **still preserves 4096 source tokens**, explicitly stops for insufficient CUDA memory, and may OOM on real 9B gradients. The T4 trial is experimental, not verified as training-capable.

[GitHub static validation #43728760228](https://github.com/kolihhan/small-agent-qlora/actions/runs/43728760228) passed, verifying all five Python cells parse, pinned model/data SHA and 4096-token no-truncation contract. **No connected Colab execution tool/session is available in this chat. No GPU started, no actual optimizer steps, no trained adapter, no model evaluation yet.** The independent original Qwen VAL benchmark remains 0/8, matched v3B GAIA100 **18/100**. A valid GPU checkpoint followed by strict held-out task comparison is necessary before any learning claim.

## PHYSICAL FIRST real Qwen3.5-9B 4K QLoRA Colab T4 adapter, 8 optimizer steps COMPLETED (2026-11-15)

Prior conclusions that NextSearch QLoRA is blocked on a suitable GPU are **superseded**: the actual existing user-owned GitHub `COLAB_ADC_JSON` refresh credentials and the official `google-colab-cli==0.8.0` supported an authenticated and **physically allocated Tesla T4 (14.56 GiB)** using [GitHub Actions run #43732429711](https://github.com/kolihhan/small-agent-qlora/actions/runs/43732429711). The workflow completed **SUCCESS**, and the physical run log contains `NEXTSEARCH_QLORA_REAL_GPU8_OPTIMIZER_PROOF` plus an independent ZIP readback `NEXTSEARCH_REAL_COLAB_QWEN9B_EIGHT_GPU_STEPS_PHYSICALLY_VERIFIED`. Original Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, no alternate base or continuation of v7/v3B; actual T4-compatible training code pinned `6465698908c6883b0f91451785caa11ebfadb090` and CUDA mixed-GEMM preflight passed.

**What physically happened:** 48 original NextSearch Apache TRAIN action windows (24 source-answer-missing → real next tool, 24 source-answer-visible → original final) across 28 distinct original source questions, all at **4096 actual Qwen token context without truncation**, exact SHA256 `5420cfdb6111806a23b4bf6cb9cf04c77c75b5f0cf4af6c0b63727f312f40432`; Qwen 4-bit NF4, LoRA on `q_proj,v_proj`, **8 actual optimizer steps**, checkpoint folders **checkpoint-4** and **checkpoint-8**, final PEFT adapter **2,232,480 bytes**, adapter SHA256 **`69deffc3ca0548778cf451332b76ee19073780d0490baba22193a36ed19c3508`**, 673 seconds spent training step run. The ZIP on Colab was 75,739,629 bytes, successfully downloaded and independently verified by Actions with `trainer_state.global_step==8`, run_config 48 examples/4096/full prompt, and matching adapter hash. [GitHub Actions artifact #14168579951](https://github.com/kolihhan/small-agent-qlora/actions/runs/43732429711/artifacts/14168579951) contains full downloaded checkpoint ZIP (GitHub artifact outer ZIP), runtime notebook output and physical report. No paid Hugging Face compute and no original v3B/v7 adapters overwritten. Experiment branch `experiment/nextsearch-colab-real-gpu8-20261115` retains the remote orchestrator and original tool data provenance.

**RESEARCH LIMIT:** This proves a durable real QLoRA adapter is trained, NOT it learned transferable tool-search behavior. Publisher source answers were not independently replayed to prove optimal next actions. No NextSearch heldout, non-GAIA OOD, or GAIA100 paired evaluation has run on this adapter yet. QLoRA v3B historical matched GAIA100 **18/100 unchanged**; report **new QLoRA adapter exists but measured improvement unknown**. Before any promotion run matched original Qwen vs new adapter on source-disjoint tasks using the actual equivalent tool schema and an unchanged GAIA control; track exact success, calls, failures, latency and over-search. Even a gain on the source VAL8 alone would be too small to establish generalization.

## REAL Qwen9B 8-step adapter vs base NextSearch VAL4 on one Colab T4 — invalid real web search environment (2026-11-16)

Physical [GitHub Actions #43737215981](https://github.com/kolihhan/small-agent-qlora/actions/runs/43737215981) **SUCCESS** completed an original Qwen3.5-9B 4bit base-vs-eight-step-PEFT paired comparison under the **same** model runtime, ABBA order, source-independent original NextSearch VAL4 (two ASearcher and two TaskCraft). Original adapter SHA `69deffc3ca0548778cf451332b76ee19073780d0490baba22193a36ed19c3508`, original frozen VAL source SHA `4cf774e6bf76210ff33cf9c41ea3374728aadf12d2d20510224b56fc116a73cd`. Strict exact **0/4→0/4**, **32 tool calls each arm**, **9 tool errors each arm**, all eight episodes hit 9-step limit and returned empty answer. However EVERY search tool observation **FAILED** (0 successful observations across all eight episodes): both actual search backends unavailable. This is an **environment-invalid evidence-based learning test**; no justified claim that the trained adapter learned or failed to learn real source use. [PR #51](https://github.com/kolihhan/small-agent-qlora/pull/51) CLOSED UNMERGED. No new adapter training, no GAIA evaluation; GAIA100 best historical **18/100 unchanged**.

To distinguish actual source use from outage, the user-authorized Colab GPU has started a NEW matched original model vs trained adapter test on **original physical 2WikiMultihopQA VALIDATION** files rather than the broken search MCP: original source frozen 2Wiki VAL8 [Actions #43712873154](https://github.com/kolihhan/small-agent-qlora/actions/runs/43712873154) cohort SHA `fded3c65e2ccc829ae0efdd8af448896320590b63bb69e0e9ff1d52eadc4fbb7`; four unchanged original independent VAL tasks, 10 real local article docs each, four annotated support docs. True first-party **local** title search and read both must pass an active tool preflight per task, before inference. Same Qwen9B 4bit original with PEFT.disable_adapter vs loaded trained8, original AgentRuntime, same prompt, 10 steps, ABBA sequence; strict normalized exact and actual source evidence/calls. [Live physical run #43744326618](https://github.com/kolihhan/small-agent-qlora/actions/runs/43744326618); [PR #52](https://github.com/kolihhan/small-agent-qlora/pull/52). No result yet as of note. Even if passing, four *public previously-evaluated* VAL questions are only a diagnostic, not pristine hidden OOD; do not promote until larger/new independent closed loop and GAIA.
