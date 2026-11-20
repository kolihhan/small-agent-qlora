"""CPU-only preregistration, strict matched scorer and privacy contract tests."""
from copy import deepcopy
import json

import pytest

from scripts.appworld_parity_preregister_new_train2 import (
    select_task_ids,freeze,require_frozen,PROTOCOL,SEED,
)
from scripts.appworld_parity_paired_fixed_qwen_new_train2 import (
    aggregate, PROTOCOL_RUN, PROTOCOL_FINAL,
)
from scripts.appworld_parity_synthetic_privacy_smoke import (
    exercise_synthetic_private_workflow, _SECRET,
)
from gaia_small_agent.tools.appworld_parity_safe_export import validate_safe_report


def rosters():
    seen=["old_a","previous_b","earlier_c","past_d",
          "pilot_e","prior_f","checked_g","known_h"]
    new=["delta_a","gamma_b","alpha_b","beta_a","alpha_a",
         "theta_a","lambda_x","mu_x","nu_x","xi_x"]
    return new+seen,seen


def test_immutable_source_only_cohort_excludes_old_eight_scenarios():
    all_ids,old=rosters()
    chosen=select_task_ids(all_ids,old)
    assert len(chosen)==2
    assert not {x.split("_")[0] for x in chosen} & {x.split("_")[0] for x in old}
    assert chosen==select_task_ids(list(reversed(all_ids)),list(reversed(old)))
    meta=freeze(all_ids,old)
    assert meta["protocol"]==PROTOCOL
    assert meta["new_scenes_disjoint_from_original_train8"]
    assert meta["n_repeats_per_arm_per_scene"]==2
    assert meta["arms"]==["baseline","parity"]
    assert meta["max_agent_steps"]==20 and meta["max_actual_api_calls"]==48
    assert meta["max_model_new_tokens_each_turn"]==384
    assert meta["model_sampling_temperature"]==0
    assert meta["no_Qwen_training_or_execution_in_this_freeze"]
    assert chosen==require_frozen(meta,all_ids,old)
    dump=json.dumps(meta)
    assert all(x not in dump for x in all_ids)
    assert "password" not in dump.lower()
    assert "access_token" not in dump


def test_frozen_source_drift_fails_closed():
    all_ids,old=rosters()
    meta=freeze(all_ids,old)
    with pytest.raises(ValueError):
        require_frozen(meta,all_ids+["newer_unknown"],old)
    with pytest.raises(ValueError):
        require_frozen({**meta,"max_agent_steps":21},all_ids,old)
    with pytest.raises(ValueError):
        require_frozen({**meta,"pinned_model_SHA256":"changed"},all_ids,old)


def row(tmp_path,scene,repeat,arm,success):
    m=exercise_synthetic_private_workflow(tmp_path/f"private-{scene}-{repeat}-{arm}")
    m["scene_index"]=scene
    m["repeat_index"]=repeat
    m["arm"]=arm
    m["official_task_success"]=bool(success)
    m["nested_state_tests"]=[int(success),3]
    m["elapsed_wall_seconds"]=100.0
    if arm=="baseline":
        for op in ("supervisor_profile","supervisor_passwords"):
            m["auth_action_counts"][op]={"attempted":0,"succeeded":0,"failed":0}
    validate_safe_report(m)
    return m


def test_strict_matched_paired_official_rescue_regression_and_instability(tmp_path):
    all_ids,old=rosters()
    frozen=freeze(all_ids,old)
    # Deliberately one rescue, one regression and two concordant pairs.
    spec=[
        {(0,"baseline"):False,(0,"parity"):True,
         (1,"baseline"):True,(1,"parity"):False},
        {(0,"baseline"):True,(0,"parity"):True,
         (1,"baseline"):False,(1,"parity"):False},
    ]
    shards=[]
    for scene in range(2):
        episodes=[row(tmp_path,scene,repeat,arm,spec[scene][repeat,arm])
                  for repeat in range(2) for arm in ("baseline","parity")]
        shards.append({"protocol":PROTOCOL_RUN,
                       "scene_index":scene,"n_episodes":4,
                       "no_private_task_or_password_or_token_or_hash_export":True,
                       "episodes":episodes})
    out=aggregate(frozen,shards)
    assert out["protocol"]==PROTOCOL_FINAL
    assert out["total_matched_pairs"]==4
    assert out["rescues"]==1 and out["regressions"]==1
    assert out["official_task_success_count"]=={"baseline":2,"parity":2}
    assert any(x["official_success_disagrees_across_repeats"] for x in out["repeat_disagreements"])
    dumped=json.dumps(out)
    assert all(secret not in dumped for secret in _SECRET)
    assert "raw_agent_trace" not in dumped


def test_incomplete_or_unmatched_episodes_are_never_aggregated(tmp_path):
    all_ids,old=rosters();frozen=freeze(all_ids,old)
    m=row(tmp_path,0,0,"parity",True)
    bad=[{"protocol":PROTOCOL_RUN,"scene_index":0,"n_episodes":4,
          "no_private_task_or_password_or_token_or_hash_export":True,
          "episodes":[m]*4},
         {"protocol":PROTOCOL_RUN,"scene_index":1,"n_episodes":4,
          "no_private_task_or_password_or_token_or_hash_export":True,
          "episodes":[m]*4}]
    with pytest.raises(ValueError):
        aggregate(frozen,bad)
