"""Physical Qwen3.5-9B same-prefix fork pilot on original non-GAIA local documents.

Original base model FIRST decides a real tool call. The identical captured
AssistantTurn (including tool call arguments/content) is replayed to both
arms, with actual original tools and a byte-identical source workspace.
The branch changes only whether a *new* tool is allowed. Scores are verified
against original synthetic source documents AFTER both answers are saved.
This is a small mechanism test, NOT GAIA and NOT a QLoRA uplift.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn
from gaia_small_agent.benchmark.fork_probe import run_pair
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools

PILOT = "real-original-qwen9b-non-gaia-tool-fork-v1"
CASES = [
    {
        "id": "finance_sufficient",
        "stratum": "evidence_sufficient",
        "question": "Read intro.txt from this workspace, then give only the FY2024 approved West regional count. Do not guess.",
        "intro": "West region FY2024 validated audit: approved count is 174. This document is the final signed record.",
        "detail": "This is an unused historical draft. Approved count before revisions was 169.",
        "answer": "174",
    },
    {
        "id": "inventory_missing",
        "stratum": "evidence_missing",
        "question": "Read intro.txt from this workspace. What is the final approved inventory count for Site R17? Return only the count.",
        "intro": "Site R17 preliminary count was 207, but it is superseded. The final certified number is in detail.txt. Do not use the preliminary number.",
        "detail": "Final signed audit, Site R17: certified inventory count is 183.",
        "answer": "183",
    },
    {
        "id": "release_sufficient",
        "stratum": "evidence_sufficient",
        "question": "Read intro.txt from this workspace, then return only the validated build identifier for Project Delta.",
        "intro": "Project Delta final validated build identifier is QX-742. This is the approved release record.",
        "detail": "Old draft, not the approved release: QX-741.",
        "answer": "QX-742",
    },
    {
        "id": "metrics_missing",
        "stratum": "evidence_missing",
        "question": "Read intro.txt from this workspace. What is the final audited November unit-cost in cents for Pilot M4? Return only the number.",
        "intro": "November unit-cost for Pilot M4: provisional report 63 cents was overturned. The signed correction is in detail.txt; don't use the provisional figure.",
        "detail": "Final audited Pilot M4 November unit-cost: 57 cents.",
        "answer": "57",
    },
]


class RecordOriginal:
    def __init__(self, base):
        self.base = base
        self.turns: list[AssistantTurn] = []

    def complete(self, messages, tools):
        turn = self.base.complete(messages, tools)
        self.turns.append(deepcopy(turn))
        return turn

    def complete_final(self, messages, tools):
        turn = self.base.complete_final(messages, tools)
        self.turns.append(deepcopy(turn))
        return turn


def _new_model(base_url: str):
    return OpenAICompatModel(model="qwen3.5-9b", base_url=base_url,
                             max_new_tokens=512, timeout_s=180.0)


def _fixture(workdir: Path, item: dict) -> None:
    workdir.mkdir(parents=True)
    (workdir / "intro.txt").write_text(item["intro"] + "\n", encoding="utf-8")
    (workdir / "detail.txt").write_text(item["detail"] + "\n", encoding="utf-8")


def execute(*, base_url: str, out_dir: Path, group: int) -> dict:
    if group not in (0, 1):
        raise ValueError("group must be 0 or 1")
    selected = [c for i, c in enumerate(CASES) if i % 2 == group]
    if out_dir.exists():
        raise FileExistsError("do not overwrite existing model evidence")
    out_dir.mkdir(parents=True)
    rows = []
    for item in selected:
        root = out_dir / item["id"]
        raw_workspace = root / "original"
        answer_workspace = root / "answer"
        cont_workspace = root / "continue"
        for path in (raw_workspace, answer_workspace, cont_workspace):
            _fixture(path, item)
        recorded = RecordOriginal(_new_model(base_url))
        original = AgentRuntime(recorded, default_tools(), max_steps=12).run(
            item["question"], raw_workspace)
        first = recorded.turns[0] if recorded.turns else None
        if not first or not first.tool_calls:
            rows.append({
                "id": item["id"], "stratum": item["stratum"],
                "status": "NO_TOOL_PREFIX_OBSERVED",
                "original_completed": original.completed,
                "original_correct": original.completed and
                                    original.answer.strip().casefold() == item["answer"].casefold(),
            })
            continue
        # Original first turn is an actual same-base model decision.
        fork = run_pair(
            item["question"], source_id=f"{PILOT}:{item['id']}",
            prefix=[first], model_factory=lambda: _new_model(base_url),
            tools_factory=default_tools,
            answer_workspace=answer_workspace,
            continue_workspace=cont_workspace,
            max_steps=12,
        )
        raw = {
            "id": item["id"], "stratum": item["stratum"], "status": "FORK_COMPLETE",
            "original_first_tool_names": [x.name for x in first.tool_calls],
            "original_correct": original.completed and
                                original.answer.strip().casefold() == item["answer"].casefold(),
            "original_steps": original.metrics.steps,
            "fork": fork,
        }
        # No answer labels in the model calls; scoring only after BOTH arms are frozen.
        for name in ("answer_now", "continue_tools"):
            arm = fork["arms"][name]
            arm["correct"] = arm["completed"] and (
                arm["answer"].strip().casefold() == item["answer"].casefold())
        rows.append(raw)
        print("FORK_PHYSICAL", item["id"], item["stratum"], "answer",
              int(fork["arms"]["answer_now"]["correct"]), "continue",
              int(fork["arms"]["continue_tools"]["correct"]), flush=True)
    complete = [r for r in rows if r["status"] == "FORK_COMPLETE"]
    result = {
        "protocol": PILOT, "fixed_model": "Qwen3.5-9B Q4_K_M",
        "same_model_tools_budgets": True, "group": group,
        "n_cases": len(rows), "n_forked": len(complete),
        "n_no_qualifying_first_tool": len(rows) - len(complete),
        "answer_now_correct": sum(int(r["fork"]["arms"]["answer_now"]["correct"]) for r in complete),
        "continue_tools_correct": sum(int(r["fork"]["arms"]["continue_tools"]["correct"]) for r in complete),
        "no_GAIA": True, "no_training": True,
        "no_new_adapter": True,
        "limitation": "Synthetic four-case local source probe. Source IDs and tool histories are actual for this run; sample is not independent natural tasks, not GAIA, and no transfer claim.",
        "cases": rows,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True)
                                        + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, type=int)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = execute(base_url=args.base_url, out_dir=args.out, group=args.group)
    print("MATCHED_REAL_QWEN9B_FORK", json.dumps({
        k: v for k, v in report.items() if k != "cases"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
