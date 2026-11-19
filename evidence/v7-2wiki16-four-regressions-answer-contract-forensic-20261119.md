# Original v7 Qwen3.5-9B: four regressions forensic, raw responses compared to publisher gold

Original real 16-case [Actions #44477751745](https://github.com/kolihhan/small-agent-qlora/actions/runs/44477751745), artifacts: four `v7-2wiki16-physical-*` predictions, blinded source, publisher references and final scored aggregate. This is **posthoc analysis on already exposed public validation**, not a new result or GAIA guarantee.

## Four old-normalized-exact regressions: 3 answer captions, 1 real incompletion
| Hashed task ID prefix | Original prompt | Evidence-prompt outcome | Root-level observable evidence |
|---|---|---|---|
| `6c7506fa08a0` | Correct `Bomgay` | **invalid_final**; empty answer; tool calls 8→16, duplicate blocked 0→3; 2→4 actual supporting docs read | **Real task completion failure**, despite all four support docs read |
| `0ed5ae7108b7` | Correct movie title | Answer ends `The final answer is: Now Where Did The 7Th Company Get To?` | Correct literal title present; old `normalize` does not strip `The ... is:` |
| `1db17ce108a7` | Correct `Ringo The Lone Rider` | Answer ends `The answer is: Ringo The Lone Rider` | Same correct literal name under unsupported old prefix |
| `033d63f308c4` | Correct `Sunday On The Rocks` | Answer ends `The final answer is: Sunday On The Rocks` | Same correct literal name under unsupported old prefix |

One other old-metric both-wrong task (`3714220108a2`) has an evidence-prompt ending `The answer is: **Yes**`, while reference is `yes`; old last-line normalizer strips an unmatched trailing markdown marker but not this introductory phrase.

### Gold-blind proposed output-contract rule
Apply **only** on the final answer line: remove anchored explicit `The answer is:`, `The final answer is:`, `Answer is:` or matching clear answer caption. No question, gold answer, retrieval, model call or intermediate reasoning is used. Unknown formats remain unchanged, including ambiguous narrative.

Diagnostic on same **already exposed** 16 tasks:
- Old frozen last-line: control **13/16**, evidence-prompt **10/16**.
- Conservative added captions: control **13/16**, evidence-prompt **14/16**.
- Evidence-prompt +4 apparent rescues, 0 formatter regressions *on exposed cases*; **not independent lift**. True `invalid_final` remains unresolved, and the “evidence prompt is better” conclusion is not established.
- This correction is not a license to rewrite the GAIA scorer. Any future deployment must feed actual cleaned final answer to the original untouched scorer; report raw and cleaned as separate metrics.

## Independent prospective screen already initiated
[Original v7 NEW public 2Wiki16 zero-extra-call output experiment](https://github.com/kolihhan/small-agent-qlora/actions/workflows/physical-v7-2wiki16-zero-call-contract.yml): **new ranked question positions 73–88**, excluding prior recorded task IDs and annotated support files; exact same original Qwen3.5-9B v7 weights, native 12-step search/read, baseline prompt, one physical inference per question. Publisher answers isolated until ALL model outputs are uploaded. Predeclared positive *screen only* = >=3 net rescues and zero regressions against frozen old normalization; actual benchmark GAIA unchanged and new QLoRA training absent.

**Separate the causes in future Small Agent policy-learning work:** (a) missing evidence, (b) wrong multihop relation / comparison, (c) actual invalid final or runaway duplicated tool calls despite evidence, and (d) final output formatting. Never optimize tool-use count or a formatted proxy instead of matched correct completions.
