from scripts.audit_2wiki_search_read_paths import replay, audit


def specimen(link=True):
    titles = ["Alpha East", "Beta West", "Gamma Hill", "Delta Court", "Unrelated"]
    sentences = [
        ["Alpha East was a notable place. " + "intro " * 30 +
         ("Beta West was the next link." if link else "No link appeared.")],
        ["Beta West was a notable place. " + "intro " * 30 + "Gamma Hill was next."],
        ["Gamma Hill had a connection. " + "intro " * 30 + "Delta Court followed."],
        ["Delta Court completed the chain. " + "intro " * 30 + "Done."],
        ["Unrelated is not cited. " + "intro " * 30],
    ]
    return {
        "id": "four-doc-example",
        "question": "What does Alpha East connect to?",
        "context": {"title": titles, "sentences": sentences},
        "supporting_facts": {"title": titles[:4], "sent_id": [0] * 4},
        "answer": "Delta Court",
    }


def test_search_read_actions_are_nonredundant_and_source_visible():
    item = replay(specimen())
    assert item["long_candidate"] is True
    assert item["visited"] == 4
    assert item["nonredundant_search_reads"] == 8
    assert item["complete"]
    assert item["action_names"] == ["search", "read"] * 4
    assert item["oracle_actions_used"] is False


def test_missing_link_blocks_long_path_despite_gold_support_labels():
    item = replay(specimen(False))
    assert item["visited"] == 1
    assert not item["long_candidate"]
    assert item["delivered_support"] == 1


def test_never_exports_training_labels():
    result = audit([specimen()], "pinned-source")
    assert result["real_content_search_read_long_corpus_candidates"] == 1
    assert result["training_labels_exported"] == 0
    assert result["model_agent_actions_tested"] == 0
    assert result["training_ready"] is False


def test_unseen_distractor_not_retrieved_just_because_in_publisher_context():
    item = replay(specimen())
    assert item["visited"] == 4
    assert item["visited"] < item["total_documents"]
