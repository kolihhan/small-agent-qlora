import pytest
from scripts.prepare_finqa_teacher_train import select_training_tasks


def row(i, report, *, answer="24.0", question=None):
    return {
        "id": str(i), "filename": report,
        "pre_text": ["Revenue for this group: 100 in 2024, 76 in 2023."],
        "post_text": ["Compare growth before any special adjustments."],
        "table": [["Metric", "2024", "2023"], ["Revenue", "100", "76"]],
        "qa": {
            "question": question or f"What was the change in revenue in report {i} from 2023 to 2024?",
            "answer": answer, "exe_ans": 24,
            "steps": [{"op": "subtract"}, {"op": "add"}],
        },
    }


def test_disjoint_finqa_train_source_and_required_tool_contract():
    train = [row(1, "doc-1"), row(2, "doc-2"), row(3, "doc-3")]
    dev = [row(9, "doc-3")]
    tasks = select_training_tasks(train, dev, limit=2)
    assert len(tasks) == 2
    assert {t["source"] for t in tasks} == {"finqa-train"}
    assert all(t["required_tools"] == ["read", "python"] for t in tasks)
    assert all("24" not in t["question"] for t in tasks)
    assert all("report.txt" in t["files"] for t in tasks)
    assert len({t["document_id_hash"] for t in tasks}) == 2


def test_reject_ambiguous_percent_and_inconsistent_number():
    rows = [row(1, "a", answer="24%"), row(2, "b", answer="19")]
    with pytest.raises(ValueError, match="insufficient"):
        select_training_tasks(rows, [], limit=1)


def test_reject_dev_question_even_when_source_document_differs():
    original = row(1, "doc-1")
    other = row(2, "doc-2", question=original["qa"]["question"])
    with pytest.raises(ValueError, match="insufficient"):
        select_training_tasks([other], [original], limit=1)
