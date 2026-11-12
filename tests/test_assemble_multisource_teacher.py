"""Source-disjoint cross-trajectory curation tests (only synthetic QA)."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from gaia_small_agent.training.protection import question_hash
from scripts.assemble_multisource_teacher import assemble
from tests.test_teacher_curation import make_row


def _make_teacher(task_id, source, question):
    row = make_row(task_id, ntools=3)
    row["source"] = source
    row["messages"][1]["content"] = question
    for index, filename in ((2, "index.txt"), (4, "article_00.txt"), (6, "article_01.txt")):
        row["messages"][index]["tool_calls"][0]["function"]["arguments"]["source"] = filename
    return row


def _setup(tmp_path, *, rows=None, forbidden=None):
    rows = rows or [
        _make_teacher("hotpot-task::sample-1", "hotpotqa-distractor-train", "Which file contains code CANON?"),
        _make_teacher("wiki-task::sample-1", "2wikimultihopqa-train", "Which source holds the archive CANON?"),
    ]
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    one.write_text(json.dumps(rows[0]) + "\n")
    two.write_text("".join(json.dumps(row) + "\n" for row in rows[1:]))
    protect = tmp_path / "gaia.json"
    protect.write_text(json.dumps({"hashes":[sha256(f"real-ish-protected-{i}".encode()).hexdigest() for i in range(165)]}))
    hold_a = tmp_path / "hotpot-holdout.json"
    hold_a.write_text(json.dumps({"hashes":[question_hash("heldout hotpot question")]}))
    hold_b = tmp_path / "wiki-holdout.json"
    hold_b.write_text(json.dumps({"hashes":[question_hash(forbidden or "heldout wiki question")]}))
    return [one,two],protect,[hold_a,hold_b]


def test_accepted_unique_real_shape_multisource_and_actions(tmp_path):
    files,protected,holdouts=_setup(tmp_path)
    rows,report=assemble(
        inputs=files,protected=protected,holdouts=holdouts,
        min_unique_tasks=2,min_tasks_per_source=1,
    )
    assert report["accepted_unique_tasks"]==2
    assert report["accepted_task_source_counts"]=={
        "2wikimultihopqa-train":1, "hotpotqa-distractor-train":1
    }
    assert report["trainable_assistant_actions"]==8
    assert all(row["training_action_indices"]==[2,4,6,8] for row in rows)
    assert report["training_ran"] is False and report["gaia_evaluation_ran"] is False
    assert all("sha256" in item for item in report["source_manifest"])


def test_insufficient_cross_source_or_task_count_fail_closed(tmp_path):
    rows=[
        _make_teacher("a::sample-1","hotpotqa-distractor-train","Independent one?"),
        _make_teacher("b::sample-1","hotpotqa-distractor-train","Independent two?"),
    ]
    files,protected,holdouts=_setup(tmp_path,rows=rows)
    with pytest.raises(ValueError,match="cross-source"):
        assemble(inputs=files,protected=protected,holdouts=holdouts,min_unique_tasks=2)
    with pytest.raises(ValueError,match="insufficient independently"):
        assemble(inputs=files,protected=protected,holdouts=holdouts,min_unique_tasks=3)


def test_protected_validation_and_duplicate_task_id_are_enforced(tmp_path):
    rows=[
        _make_teacher("a::sample-1","hotpotqa-distractor-train","Independent A"),
        _make_teacher("b::sample-1","2wikimultihopqa-train","Independent B"),
    ]
    files,protected,holdouts=_setup(tmp_path,rows=rows,forbidden="Independent B")
    with pytest.raises(ValueError,match="insufficient independently"):
        assemble(inputs=files,protected=protected,holdouts=holdouts,min_unique_tasks=2)
    holdouts[1].write_text(json.dumps({"hashes":[question_hash("other holdout")]}))
    rows[1]["task_id"]=rows[0]["task_id"]
    files[1].write_text(json.dumps(rows[1])+"\n")
    with pytest.raises(ValueError,match="duplicate"):
        assemble(inputs=files,protected=protected,holdouts=holdouts,min_unique_tasks=2)


def test_unknown_source_and_missing_protected_sets_fail(tmp_path):
    files,protected,holdouts=_setup(tmp_path)
    row=json.loads(files[1].read_text())
    row["source"]="GAIA-validation"
    files[1].write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="forbidden source"):
        assemble(inputs=files,protected=protected,holdouts=holdouts,min_unique_tasks=2)
    files[1].write_text(json.dumps(_make_teacher("wiki::sample-1", "2wikimultihopqa-train","Wiki again"))+"\n")
    protected.write_text(json.dumps({"hashes":[]}))
    with pytest.raises(ValueError,match="GAIA protection"):
        assemble(inputs=files,protected=protected,holdouts=holdouts,min_unique_tasks=2)
