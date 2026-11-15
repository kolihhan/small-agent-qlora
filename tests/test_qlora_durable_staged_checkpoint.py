from types import SimpleNamespace

import pytest
from gaia_small_agent.training.qlora import _checkpoint_stop_control, _validate_checkpoint_stop


def test_exact_stop_still_requests_trainer_checkpoint():
    assert _validate_checkpoint_stop(35, 100) == 35
    for step in (1, 34):
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        got = _checkpoint_stop_control(step, 35, control)
        assert got is control and not control.should_save and not control.should_training_stop
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    _checkpoint_stop_control(35, 35, control)
    assert control.should_save and control.should_training_stop


def test_staging_does_not_change_default_training_behavior():
    assert _validate_checkpoint_stop(None, 100) is None
    for n in (0, 100, 101, -1, True):
        with pytest.raises(ValueError):
            _validate_checkpoint_stop(n, 100)
    with pytest.raises(ValueError):
        _validate_checkpoint_stop(35, None)
