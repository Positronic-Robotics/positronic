import numpy as np
import pytest

pytest.importorskip('transformers')

from positronic.offboard.spec import Model
from positronic.policy.codecs import ACTION
from positronic.policy.processors import ChunkedSchedule
from positronic.vendors import molmoact2
from positronic.vendors.molmoact2 import starter
from positronic.vendors.molmoact2.server import molmoact2_model

CHUNK = 15


class FakeMolmoAct2(Model):
    """Answers every call with the same chunk, and keeps the observations it read."""

    def __init__(self):
        self.chunk = [np.array([i, 0, 0, 0, 0, 0, 0, i / CHUNK], dtype=np.float32) for i in range(CHUNK)]
        self.observations = []

    def __call__(self, obs, *, session_id):
        self.observations.append(obs)
        return [{ACTION: action} for action in self.chunk]


def _call(model: Model) -> list[np.ndarray]:
    obs = {molmoact2.IMAGES: [], molmoact2.STATE: np.zeros(8, dtype=np.float32), molmoact2.TASK: 'pick up the bowl.'}
    return [step[ACTION] for step in model(obs, session_id='s')]


def test_the_unedited_starter_passes_the_task_and_the_whole_chunk_through_unchanged():
    fake = FakeMolmoAct2()

    actions = _call(starter.StarterModel(fake))

    assert fake.observations[0][molmoact2.TASK] == 'pick up the bowl.'
    assert len(actions) == CHUNK
    assert all(np.array_equal(a, b) for a, b in zip(actions, fake.chunk, strict=True))


def test_the_unedited_starter_loads_the_model_as_the_example_server_does():
    defaults = molmoact2_model.kwargs

    assert starter.starter_model.kwargs == {k: defaults[k] for k in ('hf_repo', 'device_map')}
    assert starter.NUM_STEPS == defaults['num_steps']


def test_an_instruction_replaces_the_text_of_its_task_only(monkeypatch):
    monkeypatch.setattr(starter, 'INSTRUCTIONS', {'pick up the bowl.': 'grasp the red bowl'})
    fake = FakeMolmoAct2()
    model = starter.StarterModel(fake)

    _call(model)
    model({molmoact2.TASK: 'open the drawer.'}, session_id='s')

    assert [obs[molmoact2.TASK] for obs in fake.observations] == ['grasp the red bowl', 'open the drawer.']


def test_the_unedited_starter_executes_the_whole_chunk_of_each_prediction():
    assert starter.horizon_sec() is None


def test_actions_per_chunk_cuts_the_schedule_after_that_many_actions(monkeypatch):
    monkeypatch.setattr(starter, 'ACTIONS_PER_CHUNK', 4)
    schedule = ChunkedSchedule(starter.ACTION_FPS, starter.horizon_sec())

    waypoints, _ = schedule._plan_chunk([{ACTION: i} for i in range(CHUNK)], start_ns=0)

    assert [w.commands[ACTION] for w in waypoints] == [0, 1, 2, 3]


def test_a_grip_threshold_closes_the_grip_above_it_and_opens_it_below_it(monkeypatch):
    monkeypatch.setattr(starter, 'GRIP_THRESHOLD', 0.5)
    fake = FakeMolmoAct2()

    actions = _call(starter.StarterModel(fake))

    assert [a[starter.GRIP] for a in actions] == [0.0] * 8 + [1.0] * 7
    assert all(np.array_equal(a[: starter.GRIP], b[: starter.GRIP]) for a, b in zip(actions, fake.chunk, strict=True))
