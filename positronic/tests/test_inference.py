import io
import sys
from types import SimpleNamespace

import pytest

from positronic.eval import Embodiment
from positronic.inference import real


class _IdlePolicy:
    """Enough policy for the attended path to warm up and close; it is never asked for an action."""

    def __init__(self):
        self.warmed = False
        self.closed = False

    def new_session(self, *_args, **_kwargs):
        self.warmed = True
        return SimpleNamespace(close=lambda: None)

    def close(self):
        self.closed = True


def _embodiment(simulated: bool = False) -> Embodiment:
    return Embodiment(
        descriptor='stub', observations={}, commands={}, static_meta={}, meta_source=None, simulated=simulated
    )


@pytest.mark.timeout(30.0)
def test_the_keyboard_path_ends_when_the_keyboard_returns(monkeypatch):
    """How an attended run finishes: ``KeyboardControl`` returns, the world stops, ``real`` closes the policy.

    A stdin that is not a terminal is the return the test can force; ``q`` is the other one.
    """
    monkeypatch.setattr(sys, 'stdin', io.StringIO())
    policy = _IdlePolicy()

    real(policy=policy, embodiment=_embodiment(), task='stub')

    assert policy.warmed
    assert policy.closed


def test_the_keyboard_path_refuses_a_simulated_embodiment():
    """It composes a real-time world and records against the wall clock, which a simulated embodiment needs
    neither of."""
    with pytest.raises(ValueError, match='sim'):
        real(policy=_IdlePolicy(), embodiment=_embodiment(simulated=True), task='stub')
