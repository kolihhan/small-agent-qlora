import pytest

from scripts.prepare_hotpotqa_tool_tasks import convert_page, convert_row, INSTRUCTION
from gaia_small_agent.training.protection import question_hash


def make_record():
    return {
        "id": "abcdef012345",
        "question": "Which city hosted the event mentioned in the founder article?",
        "answer": "Kuala Lumpur",
        "type": "bridge", "level": "hard",
        "context": {
            "title": ["Founder biography", "Event history", "Unrelated"],
            "sentences": [
                ["The founder attended the Tech 2026 event."],
                ["The Tech 2026 event took place in Kuala Lumpur."],
                ["No relevant facts."],
            ],
        },
        "supporting_facts": {
            "title": ["Founder biography", "Event history"],
            "sent_id": [0, 0],
        },
    }


def test_hotpotqa_offline_agent_task_retains_two_hops_without_gold_in_prompt():
    record = make_record()
    task = convert_row(record, split="train")
    assert task["source"] == "hotpotqa-distractor-train"
    assert task["license"] == "CC-BY-SA-4.0"
    assert task["question"].startswith(INSTRUCTION)
    assert task["question"].endswith(record["question"])
    assert record["answer"] not in task["question"]
    assert task["expected_answer"] == record["answer"]
    assert list(task["files"]) == ["article_00.txt", "article_01.txt", "article_02.txt", "index.txt"]
    assert "article_00.txt: Founder biography" in task["files"]["index.txt"]
    assert "Tech 2026" in task["files"]["article_00.txt"]
    assert "Kuala Lumpur" in task["files"]["article_01.txt"]


def test_hotpotqa_train_validation_ids_and_hashes_are_distinct():
    record = make_record()
    tr = convert_row(record, split="train")
    va = convert_row(record, split="validation")
    assert tr["id"] != va["id"]
    assert tr["question"] == va["question"]
    assert question_hash(tr["question"]) == question_hash(va["question"])
    # Cross-split question hash filtering is still needed if data overlaps.


def test_hotpotqa_rejects_bad_multihop_and_evidence_overflow():
    record = make_record()
    record["supporting_facts"]["title"] = ["Founder biography"]
    with pytest.raises(ValueError, match="two-article"):
        convert_row(record, split="train")
    record = make_record()
    record["context"]["sentences"][0] = ["a" * 2500]
    with pytest.raises(ValueError, match="too long"):
        convert_row(record, split="train")
    record = make_record()
    record["context"]["title"][1] = record["context"]["title"][0]
    with pytest.raises(ValueError, match="two-article"):
        convert_row(record, split="train")


def test_hotpotqa_conversion_skips_bad_rows_but_fail_closed_when_none():
    good = make_record()
    bad = make_record()
    bad["context"]["sentences"][0] = ["x" * 3000]
    tasks, report = convert_page([bad, good], split="validation", limit=1)
    assert len(tasks) == 1
    assert report["accepted"] == 1
    assert report["rejected"]["article too long; do not truncate evidence"] == 1
    with pytest.raises(ValueError, match="no acceptable"):
        convert_page([bad], split="train", limit=1)
