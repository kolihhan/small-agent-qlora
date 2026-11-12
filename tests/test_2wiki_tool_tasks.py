"""Unit tests for cross-source, non-GAIA grounded multi-hop training tasks."""
import pytest

from scripts.prepare_2wiki_tool_tasks import convert_2wiki_row, select_2wiki_rows


def make_source_row(*, answer="Nova", answer_in_article=True):
    return {
        "id": "synthetic-2wiki-123",
        "question": "Which musician is linked through both pages?",
        "answer": answer,
        "type": "bridge",
        "context": {
            "title": ["Artist", "Concert"],
            "sentences": [
                ["Artist performed at the concert."],
                [("Nova performed there." if answer_in_article else "One performer played there.")],
            ],
        },
        "supporting_facts": {"title": ["Artist", "Concert"], "sent_id": [0, 0]},
    }


def test_converter_keeps_source_identity_and_never_places_answer_in_prompt():
    task = convert_2wiki_row(make_source_row(), split="train")
    assert task["id"] == "2wiki-train-synthetic-2wiki-123"
    assert task["source"] == "2wikimultihopqa-train"
    assert task["license"] == "Apache-2.0"
    assert task["generator"] == "2wiki-offline-read-agent"
    assert task["oracle_type"] == "human-annotated-exact-answer"
    assert task["expected_answer"] == "Nova"
    assert "Nova" not in task["question"]
    assert "Nova" in task["files"]["article_01.txt"]
    assert "article_00.txt" in task["files"]["index.txt"]
    assert "article_01.txt" in task["files"]["index.txt"]


def test_filter_holds_out_boolean_unobserved_or_single_document_tasks():
    with pytest.raises(ValueError, match="binary answer"):
        convert_2wiki_row(make_source_row(answer="yes"), split="train")
    with pytest.raises(ValueError, match="not present"):
        convert_2wiki_row(make_source_row(answer_in_article=False), split="train")
    invalid = make_source_row()
    invalid["supporting_facts"]["title"] = ["Artist"]
    with pytest.raises(ValueError, match="two-article"):
        convert_2wiki_row(invalid, split="train")
    with pytest.raises(ValueError, match="unsupported"):
        convert_2wiki_row(make_source_row(), split="test")


def test_page_filter_refuses_unknown_or_empty_pages():
    accepted, report = select_2wiki_rows(
        [make_source_row(answer="yes"), make_source_row()], split="train", limit=1)
    assert len(accepted) == report["tasks"] == 1
    assert report["rejected"]
    with pytest.raises(ValueError, match="no extractive"):
        select_2wiki_rows([make_source_row(answer="yes")], split="train", limit=1)
