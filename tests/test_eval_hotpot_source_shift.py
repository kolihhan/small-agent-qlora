"""Source and score invariants; CPU only, no Qwen or network."""
from __future__ import annotations

from scripts.eval_hotpot_source_shift import (
    GATE, LocalSearch, LinkedRead, PROTOCOL, aggregate, freeze, rank,
)


def row(ident: str, kind: str):
    titles = [f"City {ident}", f"Person {ident}"] + [
        f"Distractor {ident} {k}" for k in range(8)]
    support = {"title": titles[:2], "sent_id": [0, 0]}
    return {
        "id": ident, "type": kind,
        "question": f"Where was Person {ident} born?",
        "answer": f"City {ident}",
        "context": {"title": titles,
                    "sentences": [[t + " is a place.", t + " is named in the article."] for t in titles]},
        "supporting_facts": support,
    }


def test_freeze_balances_types_without_model_results():
    rows = [row(f"b-{i}", "bridge") for i in range(7)] + [
        row(f"c-{i}", "comparison") for i in range(7)]
    frozen = freeze(rows)
    assert frozen["status"] == "FROZEN_SOURCE_BEFORE_MODEL"
    assert frozen["protocol"] == PROTOCOL
    assert frozen["predeclared_gate"] == GATE
    assert frozen["groups"] == {"bridge": 4, "comparison": 4}
    assert len(frozen["cases"]) == len({c["id"] for c in frozen["cases"]}) == 8


def test_source_freeze_blocks_insufficient_or_invalid_support():
    bad = row("zero", "bridge")
    bad["supporting_facts"]["sent_id"] = [100, 0]
    try:
        freeze([bad])
    except RuntimeError as e:
        assert "could not find" in str(e)
    else:
        raise AssertionError("invalid supporting source should fail closed")


def one_case(i: int, exact_pair=(False, True), parsed_pair=(False, True),
             supports_pair=(False, True), tools_pair=(4, 4)):
    arms = {}
    for idx, arm in enumerate(("native", "linked")):
        arms[arm] = {
            "raw_exact": exact_pair[idx],
            "locked_lastline_exact": parsed_pair[idx],
            "all_two_supports_read": supports_pair[idx],
            "tool_calls": tools_pair[idx], "tool_errors": 0,
            "duplicate_calls": 0, "completed": True, "seconds": 1.0,
        }
    return {"id": f"id-{i}", "type": "bridge" if i < 4 else "comparison",
            "arms": arms}


def test_aggregate_is_paired_and_never_promotes():
    cases = [one_case(i, (False, True), (False, True), (False, True))
             if i < 2 else one_case(i, (True, True), (True, True), (True, True))
             for i in range(8)]
    shards = [{"status": "complete", "protocol": PROTOCOL,
               "model_digest": "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90",
               "source_sha": "fixed", "shard": s, "n": 2,
               "cases": cases[s*2:s*2+2]} for s in range(4)]
    result = aggregate(shards)
    assert result["raw_link_effect"]["net"] == 2
    assert result["locked_parser_link_effect"]["net"] == 2
    assert result["positive_screen_only"] is True
    assert result["promotion_allowed"] is False
    assert result["no_GAIA_tasks"] is True
    shards[3]["cases"] = shards[2]["cases"]
    try:
        aggregate(shards)
    except RuntimeError as e:
        assert "duplicate" in str(e).lower()
    else:
        raise AssertionError("duplicate cases must fail closed")


def test_local_search_no_oracle_and_extraneous_title_no_links(tmp_path):
    from gaia_small_agent.tools.base import ToolResult
    title = "Wellington"
    index = {title: "pages/wellington.txt", "Sydney": "pages/sydney.txt"}
    files = {"pages/wellington.txt": "Wellington is New Zealand's capital.",
             "pages/sydney.txt": "Sydney is in Australia."}
    search = LocalSearch(index, files)
    found = search.run({"query": "Wellington"}, tmp_path)
    assert "pages/wellington.txt" in found.content
    assert "pages/sydney.txt" not in found.content
    linked = LinkedRead(index)
    # The linked tool's non-oracle, text-only behavior is checked by direct
    # hidden-document tests in workflow; the search never uses support labels.
    assert "support" not in vars(search)
    assert "support" not in vars(linked)
