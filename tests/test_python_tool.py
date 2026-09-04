from gaia_small_agent.tools.python_tool import PythonTool


def test_python_tool_executes_expression_in_workspace(tmp_path):
    result = PythonTool(timeout_s=3).run({"code": "print(17*38)"}, tmp_path)
    assert result.ok is True
    assert result.content.strip() == "646"


def test_python_tool_rejects_imports_and_escape_surfaces(tmp_path):
    tool = PythonTool(timeout_s=3)
    cases = [
        "import os\nprint(os.environ)",
        "from pathlib import Path\nprint(Path('../secret').read_text())",
        "print(open('../secret').read())",
        "print(__import__('os').environ)",
        "print(globals())",
        "print((1).__class__)",
    ]
    for code in cases:
        result = tool.run({"code": code}, tmp_path)
        assert result.ok is False, code
        assert result.error_code == "UNSAFE_PYTHON", code


def test_python_tool_keeps_safe_local_computation(tmp_path):
    result = PythonTool(timeout_s=3).run({
        "code": "values = [3, 1, 4]\nprint(sum(values), max(values), '-'.join(str(x) for x in sorted(values)))"
    }, tmp_path)
    assert result.ok is True
    assert result.content.strip() == "8 4 1-3-4"
