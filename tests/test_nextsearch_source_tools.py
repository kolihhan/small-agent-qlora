from pathlib import Path

from gaia_small_agent.tools.base import Tool, ToolResult
from scripts.nextsearch_source_tools import ResearchSearch, ResearchFetch


class SearchFake(Tool):
    name="search"
    description=""
    schema={"type":"object"}
    def __init__(self):
        self.calls=[]
    def run(self,arguments,workspace):
        self.calls.append(arguments)
        return ToolResult(True,"real result from "+arguments["query"])


class ReadFake(Tool):
    name="read"
    description=""
    schema={"type":"object"}
    def __init__(self):
        self.calls=[]
    def run(self,arguments,workspace):
        self.calls.append(arguments)
        return ToolResult(True,"original fetched webpage visible text",
                          metadata={"offset_chars":arguments["offset_chars"]})


def test_nextsearch_search_preserves_original_call_semantics_without_gold():
    fake=SearchFake()
    search=ResearchSearch(fake)
    assert search.definition()["function"]["name"]=="search"
    result=search.run(
        {"objective":"find two facts","search_queries":["one","two"]},
        Path("/tmp")
    )
    assert result.ok and "real result from one" in result.content
    assert "real result from two" in result.content
    assert result.metadata["successful_queries"]==2
    assert fake.calls==[
        {"query":"one","max_results":5},
        {"query":"two","max_results":5}
    ]


def test_source_fetch_supports_real_observations_not_source_gold():
    fake=ReadFake()
    tool=ResearchFetch(fake)
    arg={
        "objective":"verify evidence",
        "urls":["https://example.org/a","https://example.org/b"],
        "offset":30
    }
    obs=tool.run(arg,Path("/tmp"))
    assert obs.ok and "original fetched webpage visible text" in obs.content
    assert fake.calls==[
        {"source":"https://example.org/a","offset_chars":30,"max_chars":3500},
        {"source":"https://example.org/b","offset_chars":30,"max_chars":3500}
    ]
    assert "read" not in tool.definition()["function"]["name"]


def test_bad_or_oversized_requests_are_rejected_before_side_effects():
    fake=SearchFake()
    search=ResearchSearch(fake)
    assert not search.run(
        {"objective":"hello","search_queries":[]},Path("/tmp")
    ).ok
    assert not search.run(
        {"objective":"hello","search_queries":["x"]*6},Path("/tmp")
    ).ok
    assert fake.calls==[]
    fake2=ReadFake()
    fetch=ResearchFetch(fake2)
    assert not fetch.run(
        {"objective":"read","urls":["https://example.org/a"],"offset":-2},
        Path("/tmp")
    ).ok
    assert fake2.calls==[]
