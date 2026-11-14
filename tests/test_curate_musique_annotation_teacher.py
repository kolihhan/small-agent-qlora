"""Official MuSiQue annotations may create training-only verified tool trajectories.

No BIRD, GAIA, MuSiQue DEV or earlier eight tested TRAIN parents in the
training set. These are annotation-oracle demonstrations, not model rollouts.
"""
from hashlib import sha256

from scripts.curate_musique_annotation_teacher import (
    make_oracle_demo, split_oracles, CURATED_SOURCE,
)


def _row(hop=3, kind="3hop1", parent="abc"):
    paragraphs=[{
        "idx":i,"title":f"Title {i}", "is_supporting":i<hop,
        "paragraph_text": f"Evidence article {i} describes answer{i}." if i<hop else f"Distractor passage {i} contains unrelated background text.",
    } for i in range(20)]
    q=[{
        "id":f"{parent}:{i}", "question":f"subquestion {i}?",
        "paragraph_support_idx":i,"answer":f"answer{i}"
    } for i in range(hop)]
    return {
        "id":f"{kind}__{parent}", "question":f"Find answer to question {kind} {parent}?",
        "answer":f"answer{hop-1}", "answerable":True,
        "question_decomposition":q,"paragraphs":paragraphs
    }


def test_replay_real_read_tool_and_provenance():
    row=_row()
    demo=make_oracle_demo(row)
    assert demo is not None
    assert demo["source"] == CURATED_SOURCE
    assert demo["provenance"]["generator"] == "musique-source-annotation-tool-replay"
    assert demo["selection_method"] == "human_annotated_support_paths_not_model_generated"
    assert len(demo["training_action_indices"]) == 5
    assert [demo["messages"][i]["role"] for i in demo["training_action_indices"]]==["assistant"]*5
    assert demo["messages"][-1]["content"] == "answer2"
    tool_responses=[m for m in demo["messages"] if m["role"]=="tool"]
    assert len(tool_responses)==4
    assert any("answer1" in x["content"] for x in tool_responses)
    assert demo["verification"]["final_answer_correct"] is True
    assert demo["metrics"]["tool_calls"]==4
    assert demo["derived_from_annotation_not_observed_model"] is True


def test_rejects_unsupported_or_wrong_gold_and_duplicate_support():
    bad=_row()
    bad["question_decomposition"][1]["answer"]="absent"
    assert make_oracle_demo(bad) is None
    bad=_row()
    bad["question_decomposition"][1]["paragraph_support_idx"]=0
    assert make_oracle_demo(bad) is None
    bad=_row()
    bad["answer"]="wrong-final"
    assert make_oracle_demo(bad) is None


def test_train_and_holdout_are_family_and_evidence_disjoint():
    rows=[_row(3,"3hop1",str(i)) for i in range(4)]
    rows += [_row(3,"3hop2",str(i)) for i in range(4)]
    rows += [_row(4,"4hop1",str(i)) for i in range(4)]
    rows += [_row(4,"4hop2",str(i)) for i in range(4)]
    # Prevent test fixture evidence passages being identical across parents:
    for row in rows:
        for p in row["paragraphs"]:
            p["paragraph_text"] += " " + row["id"]
    train,hold,report=split_oracles(rows,blocked_ids=set(),per_hop_train=2,per_hop_holdout=2)
    assert len(train)==4 and len(hold)==4
    assert report["gaia_training_rows"]==0
    assert report["train_composition_families"]==["3hop1","4hop1"]
    assert report["heldout_composition_families"]==["3hop2","4hop2"]
    assert not {d["task_id"] for d in train} & {d["task_id"] for d in hold}


def test_previous_8_case_ids_excluded_and_not_fabricated():
    row=_row()
    excluded={sha256(row["id"].encode()).hexdigest()}
    train,hold,report=split_oracles([row],blocked_ids=excluded,per_hop_train=1,per_hop_holdout=1,require_full=False)
    assert not train and not hold
    assert report["excluded_previous_evaluation"]==1
