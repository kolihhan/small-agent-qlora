"""Research-only prefix labels from independent official task-state replays.

No text answer, reference string, GAIA result or teacher-forced next-token
accuracy is consulted. Source receipts must be validated by the data producer;
this gate cannot verify a self-reported receipt's signature on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


@dataclass(frozen=True)
class PrefixEvidence:
    source: str
    split: str
    steps: int
    prefix_step: int
    task_family: str
    source_sha256: str
    trajectory_sha256: str
    provenance_checked: bool
    task_family_disjoint: bool
    official_prefix_success: bool | None
    official_final_success: bool | None
    prefix_replayed: bool
    final_replayed: bool
    same_environment: bool
    next_action_verified: bool


@dataclass(frozen=True)
class PrefixLabel:
    label: Literal["STOP", "CONTINUE", "UNKNOWN"]
    reason: str


def label_prefix(row: PrefixEvidence) -> PrefixLabel:
    """Label only state-replayed long-horizon *training* prefixes.

    A failing final does not establish a missing STOP/CONTINUE decision.
    CONTINUE is justified only by subsequent independently successful
    closed-loop continuation in the same environment.
    """
    unknown = lambda reason: PrefixLabel("UNKNOWN", reason)
    if row.source.casefold() != "appworld" or row.split != "train":
        return unknown("nontrain_or_nonstateful_official_source")
    if not row.task_family or not row.task_family_disjoint:
        return unknown("source_or_family_not_disjoint")
    if not row.provenance_checked or not all(
        re.fullmatch(r"[a-f0-9]{64}", s) for s in (row.source_sha256, row.trajectory_sha256)
    ):
        return unknown("missing_attested_provenance")
    if (type(row.steps) is not int or type(row.prefix_step) is not int
            or row.steps < 7 or not 1 <= row.prefix_step <= row.steps):
        return unknown("not_verified_long_horizon_prefix")
    if not (row.prefix_replayed and row.final_replayed and row.same_environment):
        return unknown("missing_independent_paired_state_replay")
    if (type(row.official_prefix_success) is not bool
            or type(row.official_final_success) is not bool):
        return unknown("invalid_or_missing_official_result")
    if row.official_prefix_success and row.official_final_success:
        return PrefixLabel("STOP", "official_task_already_successful_at_prefix")
    if (not row.official_prefix_success and row.official_final_success
            and row.prefix_step < row.steps and row.next_action_verified):
        return PrefixLabel("CONTINUE", "official_successful_replayed_continuation")
    return unknown("no_verified_successful_continuation")
