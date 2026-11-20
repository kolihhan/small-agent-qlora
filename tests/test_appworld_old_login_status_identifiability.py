"""Synthetic historical trace identifiability: never make up API identities."""
from scripts.appworld_old_login_status_identifiability import inspect


def fake(nlogin=8):
    status=[401,422]+[401]*10
    steps=[3,7,*range(10,20)]
    return {
        "protocol":"APPWORLD-QWEN35-TRAIN2-TOKEN-PROVENANCE-AND-POST401-RECOVERY-V1",
        "scene_index":1,
        "actual_http_errors":12,"actual_api_calls":12,
        "observed_error_counts":{"HTTP_ERROR":12},
        "http_error_events":[{"api_call_index":i+1,"step":s,"status":h}
                             for i,(s,h) in enumerate(zip(steps,status))],
        "abstract_tool_trajectory":[{"tool":"appworld_call","outcome":"HTTP_ERROR","step":s}
                                    for s in steps],
        "first_401_token_provenance_and_post_failure_recovery":{
            "first401_present":True,"eligible_for_provenance":True,
            "first401_step":3,"post401_login_attempted":bool(nlogin),
            "post401_login_attempt_count":nlogin,
            "token_value_or_hash_exported":False,
        }
    }


def test_historical_login_status_has_bounds_not_invented_identity():
    d=inspect(fake())
    assert d["post401_login_attempts_observed_in_private_runner"]==8
    assert d["post401_API_error_calls"]==11
    assert d["post401_HTTP_status_distribution"]=={"401":10,"422":1}
    assert d["possible_login_HTTP_status_count_bounds"]=={"401":[7,8],"422":[0,1]}
    assert d["exact_per_login_status_join_from_retained_artifact"] is False
    assert d["do_not_assert_422_was_login"] is True
    assert "password" not in repr(d)
    assert "access_token" not in repr(d)


def test_requires_full_call_level_reachability():
    old=fake();old["actual_api_calls"]=11
    try:inspect(old)
    except AssertionError:pass
    else:raise AssertionError("must fail closed if one login call could be blocked")


def test_no_login_gives_exact_zero_possible_login_statuses():
    d=inspect(fake(nlogin=0))
    assert d["possible_login_HTTP_status_count_bounds"]=={"401":[0,0],"422":[0,0]}
