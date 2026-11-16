from __future__ import annotations

import pytest

from gaia_small_agent.benchmark.scoring import gaia_score
from scripts.audit_2wiki_answer_format_confound import last_line_answer


@pytest.mark.parametrize(
    ("original","parsed"),
    [
        ("Roy Del Ruth died earlier.\n\n**What Price Hollywood?**","What Price Hollywood?"),
        ("Florestano Vancini was born earlier.\n\n**Seasons Of Our Love**","Seasons Of Our Love"),
        ("The director was older.\n\nAnswer: This Love Of Ours","This Love Of Ours"),
        ("The director was older.\n\nAnswer:   Tangled Destinies","Tangled Destinies"),
        ("Ghost-Town Gold","Ghost-Town Gold"),
        ("Answer: $1,200","$1,200"),
        ("No verifiable answer", "No verifiable answer"),
        ("", ""),
    ],
)
def test_gold_blind_last_line_parser(original,parsed):
    assert last_line_answer(original)==parsed


def test_parser_never_searches_whole_explanation_for_target():
    # An explanation may mention both candidates, but the last line is wrong.
    generated="The evidence mentions A and B.\n\nAnswer: B"
    assert not gaia_score(last_line_answer(generated),"A")


def test_original_score_is_unchanged_without_mandated_parser():
    raw="Older director was born in 1926.\n\n**Seasons Of Our Love**"
    assert gaia_score(raw,"Seasons Of Our Love") is False
    assert gaia_score(last_line_answer(raw),"Seasons Of Our Love") is True
    assert last_line_answer(None) is None
