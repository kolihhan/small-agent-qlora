"""Synthetic pinned-Supervisor credential accessibility audit, no real secrets."""
from types import SimpleNamespace

from scripts.appworld_train_supervisor_credential_capability import inspect,summarize


def fake_world():
    spec=lambda params:{"parameters":params,"description":"Public spec only"}
    supervisor={
        "show_profile":spec([]),
        "show_account_passwords":spec([]),
        "complete_task":spec([]),
    }
    login=spec([
        {"name":"username","type":"string","required":True},
        {"name":"password","type":"string","required":True},
    ])
    task=SimpleNamespace(
        allowed_apps=["supervisor","calendar","api_docs","admin"],
        api_docs={"supervisor":supervisor,"calendar":{
            "login":login,
            "read_data":spec([{"name":"access_token","required":True,"type":"string"}])}},
        instruction="Update a calendar event with a known title.",
        supervisor=SimpleNamespace(email="REDACT_PRIVATE_ACCOUNT_EMAIL"),
    )
    return SimpleNamespace(task=task)


def test_official_credential_lookup_is_documented_but_bridge_denied():
    out=inspect(fake_world())
    assert out["standard_credential_retrieval_capability"]=="OFFICIAL_CREDENTIAL_LOOKUP_DOCUMENTED_BUT_BRIDGE_BLOCKED"
    assert out["public_supervisor_credential_APIs_both_documented"] is True
    assert out["both_credential_APIs_callable_via_current_bridge"] is False
    assert out["supervisor_complete_task_callable"] is True
    assert out["model_docs_lists_supervisor_completion"] is True
    assert out["model_docs_lists_credential_APIs"] is False
    assert out["original_prompt_contains_task_instruction"] is True
    assert out["original_prompt_contains_supervisor_email_value"] is False
    assert out["public_login_app_count"]==1
    assert out["public_login_required_parameter_count"]==2
    assert out["alternate_legal_credentials_from_instruction_or_public_APIs"]=="UNDETERMINED"
    assert out["credentials_validity_or_account_scope"]=="UNDETERMINED"
    assert "REDACT_PRIVATE_ACCOUNT_EMAIL" not in repr(out)
    assert "show_account_passwords" not in repr(out)
    assert "username" not in repr(out)
    assert "password" not in repr(out)


def test_missing_public_supervisor_docs_never_claims_sufficient_credentials():
    world=fake_world()
    world.task.api_docs["supervisor"].pop("show_account_passwords")
    out=inspect(world)
    assert out["standard_credential_retrieval_capability"]=="PUBLIC_SUPERVISOR_CONTRACT_INCOMPLETE"
    assert out["alternate_legal_credentials_from_instruction_or_public_APIs"]=="UNDETERMINED"


def test_eight_sources_are_safe_counts_only_and_no_real_api_calls():
    world=fake_world()
    rows=[{"scene_index":i,"safe":inspect(world)} for i in range(8)]
    out=summarize(rows)
    assert out["main_research_decision"]=="BRIDGE_CAPABILITY_MISSING_BEFORE_AUTH_PLANNING"
    assert out["source_scenes"]==8
    assert out["unique_public_contract_profiles"]==1
    assert out["no_Qwen_calls"] and out["no_login_attempt"]
    assert out["can_all_user_account_credentials_be_obtained_legitimately"]=="UNDETERMINED"
    assert "REDACT_PRIVATE_ACCOUNT_EMAIL" not in repr(out)
