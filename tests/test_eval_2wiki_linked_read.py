from pathlib import Path

from scripts.eval_2wiki_linked_read import (
    LinkedRead, EXPOSED_IDS, EXPOSED_SUPPORT_PATHS, freeze
)
from scripts.eval_2wiki_native_agent import page_path
from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS


def example(i):
    titles = [f"Movie{i}", f"Person{i}", f"Third{i}", f"Fourth{i}", f"Decoy{i}", f"Spare{i}"]
    texts = [
        f"Movie{i} was directed by Person{i}.", f"Person{i} was known for Fourth{i}.",
        f"Third{i} relates to Fourth{i}.", f"Fourth{i} finished.",
        f"Decoy{i} is unrelated.", f"Spare{i}."
    ]
    return {
        "id": "new-case-" + str(i),
        "question": f"Who is Person{i}?",
        "answer": f"Person{i}",
        "context": {"title": titles, "sentences": [[t] for t in texts]},
        "supporting_facts": {"title": titles[:4], "sent_id": [0, 0, 0, 0]},
    }


def test_link_requires_original_source_visible_title_and_real_page(tmp_path):
    indexed = {
        "Movie Alpha": page_path("Movie Alpha"),
        "Person Bravo": page_path("Person Bravo"),
        "Unrelated Charlie": page_path("Unrelated Charlie")
    }
    for title, path in indexed.items():
        f = tmp_path / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            ("Movie Alpha was directed by Person Bravo, a filmmaker born in 1900.\n"
             + "interesting " * 1000) if title == "Movie Alpha" else "hello"
        )
    tool = LinkedRead(indexed)
    result = tool.run({"source": indexed["Movie Alpha"]}, tmp_path)
    assert result.ok
    assert indexed["Person Bravo"] in result.metadata["related_local_source_paths"][0]
    assert indexed["Unrelated Charlie"] not in str(result.metadata)
    assert result.content.startswith("Movie Alpha was")
    bounded, meta = result.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
    assert len(bounded) <= MAX_TOOL_OBSERVATION_CHARS
    assert indexed["Person Bravo"] in bounded
    assert isinstance(meta["next_offset_chars"], int)


def test_nonexistent_path_and_unmentioned_author_never_disclosed(tmp_path):
    path = page_path("Movie Alpha")
    (tmp_path / "pages").mkdir(parents=True)
    (tmp_path / path).write_text("Movie Alpha made a film with no specific person's name.")
    indexed = {"Movie Alpha": path, "Person Bravo": page_path("Person Bravo")}
    tool = LinkedRead(indexed)
    result = tool.run({"source": path}, tmp_path)
    assert result.ok
    assert "related_local_source_paths" not in result.metadata
    assert not tool.run({"source": "pages/unavailable.txt"}, tmp_path).ok


def test_no_prior_id_or_prior_support_page_leaks_to_frozen_cohort():
    rows = [example(i) for i in range(8)]
    frozen = freeze(rows)
    assert len(frozen["cases"]) == 8
    assert not {c["id"] for c in frozen["cases"]} & EXPOSED_IDS
    assert not {p.split("/")[-1][:-4] for c in frozen["cases"] for p in c["support"]} & EXPOSED_SUPPORT_PATHS
    assert all(len(c["support"]) == 4 and len(c["files"]) == 6 for c in frozen["cases"])


def test_no_oracle_facts_from_support_labels_in_linking(tmp_path):
    indexed = {"Movie Alpha": page_path("Movie Alpha"), "Person Bravo": page_path("Person Bravo")}
    (tmp_path / "pages").mkdir()
    (tmp_path / indexed["Movie Alpha"]).write_text("Movie Alpha directed by an unnamed person.")
    result = LinkedRead(indexed).run({"source": indexed["Movie Alpha"]}, tmp_path)
    assert "related_local_source_paths" not in result.metadata
