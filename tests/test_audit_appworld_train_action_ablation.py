"""Safe CPU predicate tests; no official AppWorld data or simulator needed."""
import pytest
from scripts import audit_appworld_train_action_ablation as a

def _req(url,method="get"):
    return {"method":method,"url":url,"data":{}}

def test_choose_late_mutation_not_auth_or_supervisor_or_search():
    p=[_req("/spotify/read") for _ in range(7)]
    p += [_req("/spotify/login","post"),
          _req("/supervisor/complete_task","post"),
          _req("/venmo/send_payment","post"),
          _req("/spotify/search_tracks","post"),
          _req("/venmo/update_balance","patch")]
    assert a.late_candidate(p)==11

def test_no_early_post_or_safety_ignored_counted_as_mutation():
    x=[_req("/spotify/write","post") for _ in range(6)]
    x += [_req("/supervisor/complete_task","post"),
          _req("/spotify/login","post"),
          _req("/spotify/search_items","post")]
    assert a.late_candidate(x) is None

def test_source_selection_contract_is_train_and_no_student():
    assert "TRAIN" in a.PROTOCOL
    assert "abl" in a.PROTOCOL.lower()
    assert len(a.SEED)>16
