"""Strict source-safe post-401 login/status bounds from retained original traces.

The historical artifact has numeric status and model step but explicitly
omits per-call API identity. Never infer a one-to-one login->HTTP mapping.
Compute *sharp feasible bounds* and exact unidentifiability, no Qwen.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

PROTOCOL="APPWORLD-TRAIN2-OLD-TRACE-LOGIN-STATUS-IDENTIFIABILITY-V1"


def inspect(original:dict)->dict:
    assert original["protocol"]=="APPWORLD-QWEN35-TRAIN2-TOKEN-PROVENANCE-AND-POST401-RECOVERY-V1"
    p=original["first_401_token_provenance_and_post_failure_recovery"]
    assert p["eligible_for_provenance"] and p["first401_present"]
    assert p["token_value_or_hash_exported"] is False
    first=p["first401_step"]
    events=original["http_error_events"]
    assert len(events)==original["actual_http_errors"]
    ids=[x["api_call_index"] for x in events]
    assert sorted(ids)==list(range(1,len(events)+1))
    assert all(isinstance(x["status"],int) and isinstance(x["step"],int) for x in events)
    first_http=[x for x in events if x["step"]==first and x["status"]==401]
    assert len(first_http)==1
    after=[x for x in events if x["api_call_index"]>first_http[0]["api_call_index"]]
    hist=Counter(x["status"] for x in after)
    attempted=p["post401_login_attempt_count"]
    assert p["post401_login_attempted"] is bool(attempted)
    assert 0<=attempted<=len(after)
    # Every recorded login tool call reached requester iff there is exactly
    # one real HTTP event for every appworld_call result in this cohort.
    failed_calls=[x for x in original["abstract_tool_trajectory"]
                  if x["tool"]=="appworld_call" and x["outcome"]=="HTTP_ERROR"]
    all_login_statuses_confirmed=(
        len(failed_calls)==original["actual_api_calls"]==len(events)
        and original["observed_error_counts"].get("DUPLICATE_CALL",0)==0)
    assert all_login_statuses_confirmed, "not all login actions provably reached API"
    # Pigeonhole bounds: of L login failures among N remaining error events,
    # status class S with n_S occurrences -> min max(0,L-(N-n_S)).
    # This is NOT per-call attribution. Future private event IDs needed.
    bounds={}
    for status in sorted(hist):
        n=hist[status]
        bounds[str(status)]=[max(0,attempted-(len(after)-n)), min(attempted,n)]
    exact_identity_present=all(
       isinstance(x.get("api_call_id"),str) and isinstance(x.get("api_name"),str)
       for x in events)
    assert not exact_identity_present, "this audit expects no per-event API identifiers"
    assert sum(hist.values())==len(after)
    return {
        "protocol":PROTOCOL,
        "scene_index":original["scene_index"],
        "post401_API_error_calls":len(after),
        "post401_login_attempts_observed_in_private_runner":attempted,
        "all_observed_login_calls_reached_requester":all_login_statuses_confirmed,
        "post401_HTTP_status_distribution":dict(sorted((str(k),v) for k,v in hist.items())),
        "possible_login_HTTP_status_count_bounds":bounds,
        "exact_per_login_status_join_from_retained_artifact":False,
        "contains_public_login_api_identity_or_call_argument_keys":False,
        "has_login_parameter_contract_or_credential_validity_proof":False,
        "old_raw_trace_not_recoverable_from_this_artifact":True,
        "no_new_Qwen_calls":True,
        "do_not_assert_422_was_login":True,
        "requires_private_call_level_instrumentation_for_exact_attribution":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable output exists")
    d=inspect(json.loads(a.source.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
    print("EXACT_LOGIN_HTTP_JOIN_UNIDENTIFIABLE",json.dumps(d,sort_keys=True),flush=True)


if __name__=="__main__":main()
