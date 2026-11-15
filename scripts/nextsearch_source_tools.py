"""Experimental real-web tool adapter for original NextSearch search/fetch schemas.

Existing GAIA SearchTool/ReadTool implementations remain UNCHANGED.
The source dataset calls search(objective, search_queries) and
fetch(objective, urls, offset). We expose exactly those source argument
keys in an isolated harness, delegating every output to a real native
search/read invocation. No cached gold answers, fabricated pages or
publisher tool observations are accessible to the agent.

This adapter is for paired baseline/evaluation under the *same* tool
interface; not production GAIA policy deployment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gaia_small_agent.tools.base import Tool, ToolResult
from gaia_small_agent.tools.search import SearchTool
from gaia_small_agent.tools.read import ReadTool


class ResearchSearch(Tool):
    name="search"
    description=(
        "Search the public web for factual sources with several relevant "
        "queries. Use retrieved citations to find the requested answer."
    )
    schema={
        "type":"object",
        "properties":{
            "objective":{"type":"string","minLength":1},
            "search_queries":{
                "type":"array","items":{"type":"string","minLength":1},
                "minItems":1,"maxItems":5
            },
        },
        "required":["objective","search_queries"],
        "additionalProperties":False,
    }

    def __init__(self, underlying:Tool|None=None):
        self.search=underlying if underlying is not None else SearchTool()

    def run(self, arguments:dict[str,Any], workspace:Path)->ToolResult:
        error=self.validate_arguments(arguments)
        if error:
            return ToolResult(False,error,"BAD_ARGUMENTS")
        outputs=[]
        okay=0
        for idx,query in enumerate(arguments["search_queries"]):
            record=self.search.run({"query":query,"max_results":5},workspace)
            if record.ok:
                okay+=1
            outputs.append(
                f"Search {idx+1}: {query}\n"+
                (record.observation() if record.ok else
                 f"Search failed [{record.error_code or 'SEARCH_ERROR'}]: {record.content}")
            )
        return ToolResult(
            bool(okay),"\n\n".join(outputs),
            None if okay else "SEARCH_ERROR",
            {"requested_queries":len(arguments["search_queries"]),
             "successful_queries":okay}
        )


class ResearchFetch(Tool):
    name="fetch"
    description=(
        "Retrieve public web pages previously identified during search; "
        "supports multiple URLs and a resume offset."
    )
    schema={
        "type":"object",
        "properties":{
            "objective":{"type":"string","minLength":1},
            "urls":{
                "type":"array","items":{"type":"string","format":"uri"},
                "minItems":1,"maxItems":5
            },
            "offset":{"type":"integer","minimum":0}
        },
        "required":["objective","urls"],
        "additionalProperties":False
    }

    def __init__(self,underlying:Tool|None=None):
        self.reader=underlying if underlying is not None else ReadTool()

    def run(self, arguments:dict[str,Any], workspace:Path)->ToolResult:
        error=self.validate_arguments(arguments)
        if error:
            return ToolResult(False,error,"BAD_ARGUMENTS")
        blocks=[]
        okay=0
        for idx,url in enumerate(arguments["urls"]):
            result=self.reader.run(
                {"source":url,"offset_chars":arguments.get("offset",0),"max_chars":3500},
                workspace
            )
            if result.ok:
                okay+=1
            blocks.append(
                f"Fetch {idx+1}: {url}\n" +
                (result.observation() if result.ok
                 else f"Fetch failed [{result.error_code or 'FETCH_ERROR'}]: {result.content}")
            )
        return ToolResult(
            bool(okay),"\n\n".join(blocks),
            None if okay else "FETCH_ERROR",
            {"requested_urls":len(arguments["urls"]),
             "successful_fetches":okay}
        )
