from gaia_small_agent.agent.loop import SYSTEM_PROMPT


def test_system_prompt_tracks_task_goal_without_fixed_workflow():
    prompt = SYSTEM_PROMPT.lower()

    assert "what exactly" in prompt
    assert "still missing" in prompt
    assert "original task" in prompt
    assert "no fixed workflow" in prompt
    assert "do not reveal hidden chain-of-thought" in prompt


def test_system_prompt_requires_strategy_change_after_unproductive_action():
    prompt = SYSTEM_PROMPT.lower()

    assert "fails or repeats" in prompt
    assert "change strategy" in prompt
    assert "useful new evidence" in prompt
