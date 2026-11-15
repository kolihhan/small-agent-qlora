from scripts.audit_2wiki_train_visible_evidence import audit, inspect_row, canonical_row


def row(first_sentence="Alpha page cites Beta page."):
    return {
        "id": "independent-a",
        "question": "What is the relation between Alpha page and its author?",
        "context": {
            "title": ["Alpha page", "Beta page", "Distractor"],
            "sentences": [
                [first_sentence],
                ["Beta page was authored by L. Example."],
                ["Irrelevant"],
            ],
        },
        "supporting_facts": {
            "title": ["Alpha page", "Beta page"],
            "sent_id": [0, 0],
        },
        "answer": "L. Example",
    }


def test_reachability_requires_visible_link_not_gold_only():
    good = inspect_row(row())
    bad = inspect_row(row("Alpha has an unknown author."))
    assert good["reachable"] == 2
    assert good["all_support_visible"] is True
    assert bad["reachable"] == 1
    assert bad["all_support_visible"] is False


def test_reachable_but_truncated_support_must_not_be_labeled_complete():
    a = row()
    a["context"]["sentences"][0] = ["A" * 2600, "Alpha page cites Beta page."]
    a["supporting_facts"]["sent_id"][0] = 1
    result = inspect_row(a)
    assert result["all_support_visible"] is False
    assert result["delivered_support_documents"] == 0


def test_missing_source_support_rejected_not_exported():
    a = row()
    a["supporting_facts"]["title"][1] = "unavailable"
    r = audit([a], "pinned-revision")
    assert len(r["invalid_cases"]) == 1 and r["training_labels_exported"] == 0
    assert r["training_ready"] is False


def test_only_true_source_chain_distributions_reported():
    r = audit([row()], "pinned-revision")
    assert r["support_documents_distribution"] == {2: 1}
    assert r["long_ge6_distinct_support_docs_and_visible"] == 0
    assert r["decision"] == "SOURCE_INSUFFICIENT"
