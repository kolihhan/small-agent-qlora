"""Pure synthetic public docs tests; do not use protected AppWorld tasks."""
from types import SimpleNamespace

from scripts.appworld_train_public_auth_visibility_census import source_only_auth_census


def fake_world():
    long_description="A public docs description. "*100
    def api(params,desc="short public docs"):
        return {"description":desc,"method":"GET","path":"/synthetic",
                "parameters":params}
    token={"name":"access_token","required":True,"description":"Real token",
           "type":"string"}
    optional=dict(token,required=False)
    return SimpleNamespace(task=SimpleNamespace(
        allowed_apps=["calendar","api_docs","admin"],
        api_docs={"calendar":{
            "login":api([]),
            "protected_visible":api([token]),
            "protected_hidden":api([{"name":"content","required":True,
                                     "type":"string","description":long_description},token]),
            "optional_public":api([optional]),
            "no_auth":api([{"name":"id","required":True,"type":"integer",
                           "description":"ID"}]),
        }}))


def test_public_required_auth_field_can_be_present_in_schema_but_clipped():
    out=source_only_auth_census(fake_world())
    assert out["public_api_docs"]==5
    assert out["required_token_api_count"]==2
    assert out["required_token_apis_with_public_login"]==2
    assert out["required_token_visible_to_model"]==1
    assert out["required_token_missing_in_model_view"]==1
    assert out["required_token_doc_truncated"]==1
    assert out["optional_token_api_count"]==1
    assert out["optional_token_visible_to_model"]==1
    assert out.get("required_token_no_login_in_same_app",0)==0
    assert "public_apps" in out and out["public_apps"]==1
    assert "access_token" not in repr(out)
