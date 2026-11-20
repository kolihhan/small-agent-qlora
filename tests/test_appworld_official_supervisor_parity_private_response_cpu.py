"""No-Qwen fake private Supervisor response checks, never print sensitive content."""
from types import SimpleNamespace

from scripts.appworld_official_supervisor_parity_private_response_cpu import (
    audit_world, primitive_structure_match,
)

SECRET_PASSWORD="SYNTHETIC_NEVER_LOG_THIS_PASSWORD_987654"
SECRET_EMAIL="SYNTHETIC_NEVER_LOG_THIS_EMAIL@example.invalid"


class Requester:
    def __init__(self):
        self.names=[]
    def request(self,**kw):
        self.names.append(kw["_api_name"])
        if kw["_api_name"]=="show_profile":
            return {"email":SECRET_EMAIL,"first_name":"Fake"}
        if kw["_api_name"]=="show_account_passwords":
            return [{"account_name":"mail","password":SECRET_PASSWORD}]
        raise RuntimeError("API should never be called")


def world():
    def public(success):
        return {"description":"Public fake API docs","method":"GET",
                "path":"/simulator","parameters":[],
                "response_schemas":{"success":success}}
    return SimpleNamespace(
        task=SimpleNamespace(
            allowed_apps=["supervisor","mail"],
            api_docs={"supervisor":{
                "complete_task":public({"success":True}),
                "show_profile":public({"email":"sample@example.invalid","first_name":"Fake"}),
                "show_account_passwords":public([{"account_name":"mail","password":"samplepassword"}]),
            }},instruction="A generic task."
        ),requester=Requester()
    )


def test_real_private_read_only_response_probe_returns_only_safe_bools():
    w=world()
    result=audit_world(w)
    assert w.requester.names==["show_profile","show_account_passwords"]
    assert result["official_lookup_read_only_calls"]==2
    assert result["both_successful"]
    assert all(v["top_level_matches_official_public_example_shape"]
               and v["overlapping_official_public_fields_have_compatible_types"]
               and v["model_visible_2k_observation_is_valid_JSON"]
               for v in result["private_response_contract_only"])
    assert SECRET_PASSWORD not in repr(result)
    assert SECRET_EMAIL not in repr(result)
    assert "samplepassword" not in repr(result)
    assert "sample@example.invalid" not in repr(result)
    assert result["no_app_login_or_supervisor_write_calls"]
    assert result["Qwen_calls"]==0 and result["no_training"]


def test_mismatched_structure_detected_without_value_export():
    assert primitive_structure_match([{"password":"not_real"}],[{"password":123}]) is False
    assert primitive_structure_match({"email":"not_real"},{"email":SECRET_EMAIL}) is True
    assert primitive_structure_match([{"password":"not_real"}],{"password":SECRET_PASSWORD}) is False
