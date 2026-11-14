import pytest

from scripts.prepare_musique_multihop_train import select_training_tasks


def row(name, hops=3, *, answerable=True):
    return {
        "id": f"{hops}hop__{name}",
        "answerable": answerable,
        "question": f"Which artist influenced the album in scenario {name}?",
        "answer": f"Artist {name}",
        "question_decomposition": [{"id": i, "question": f"subquestion {i}"} for i in range(hops)],
        "paragraphs": [
            {"idx": i, "title": f"Document {name} {i}",
             "paragraph_text": f"Document text for {name} {i}. The artist appears later.",
             "is_supporting": i < hops}
            for i in range(20)
        ],
    }


def test_musique_source_is_genuine_three_four_hop_with_full_distractors():
    train = [row("a", 3), row("b", 4), row("c", 2), row("d", 3, answerable=False)]
    selected = select_training_tasks(train, [], limit=2)
    assert {t["hop_count"] for t in selected} == {3, 4}
    assert all(len(t["files"]) == 21 for t in selected)
    assert all(t["required_tools"] == ["read"] for t in selected)
    assert all("passages/index.txt" in t["files"] for t in selected)
    assert all("is_supporting" not in str(t["files"]) for t in selected)


def test_exclude_dev_duplicate_and_bad_supporting_evidence():
    bad = row("bad", 3)
    bad["paragraphs"][2]["is_supporting"] = False
    with pytest.raises(ValueError, match="insufficient"):
        select_training_tasks([row("dev", 3), bad], [row("dev", 3)], limit=2)
