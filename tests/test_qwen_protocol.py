from gaia_small_agent.model.transformers_qwen import parse_qwen_turn


def test_parse_native_qwen_tool_call():
    text = "<tool_call>\n<function=python>\n<parameter=code>\nprint(6*7)\n</parameter>\n</function>\n</tool_call>"
    turn = parse_qwen_turn(text)
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "python"
    assert turn.tool_calls[0].arguments == {"code": "print(6*7)"}


def test_parse_native_qwen_tool_call_accepts_json_parameter():
    text = '<tool_call>\n<function=inspect>\n<parameter=path>\n"report.xlsx"\n</parameter>\n</function>\n</tool_call>'
    turn = parse_qwen_turn(text)
    assert turn.tool_calls[0].arguments == {"path": "report.xlsx"}


def test_parse_final_text_removes_think_block():
    turn = parse_qwen_turn("<think>private reasoning</think>\n\n42")
    assert turn.content == "42"
    assert turn.tool_calls == []
