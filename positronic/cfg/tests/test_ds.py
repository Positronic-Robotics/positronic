from pathlib import Path
from typing import Any

from positronic import keys
from positronic.cfg.ds import teleop_demos
from positronic.dataset import Time
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.recording import HARNESS_WORLD
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd

ATTEMPT_START = f'{policy_keys.POLICY_META}.{policy_keys.ATTEMPT_START_NS}'


def _record(root: Path, *statics: dict[str, Any]) -> LocalDataset:
    """One episode for each item of `statics`, with grips at 1000, 2000 and 3000 ns."""
    with LocalDatasetWriter(root) as writer:
        for items in statics:
            with writer.new_episode() as episode:
                for ts in (1000, 2000, 3000):
                    episode.append(keys.GRIP, float(ts), Time(**{HARNESS_WORLD: ts}))
                for name, value in items.items():
                    episode.set_static(name, value)
    return LocalDataset(root)


def test_teleop_demos_keep_the_episodes_that_did_the_task_after_start_task(tmp_path):
    recorded = _record(
        tmp_path,
        {keys.TASK: 'complete', eval_keys.POLICY_END: PolicyEnd.COMPLETE.value, ATTEMPT_START: 1500},
        {keys.TASK: 'give up', eval_keys.POLICY_END: PolicyEnd.GIVE_UP.value, ATTEMPT_START: 1500},
        {keys.TASK: 'success', eval_keys.SUCCESS: True, ATTEMPT_START: 1500},
        {keys.TASK: 'failure', eval_keys.SUCCESS: False, ATTEMPT_START: 1500},
        {keys.TASK: 'time limit', eval_keys.TERMINATED: False, ATTEMPT_START: 1500},
        {keys.TASK: 'success before start task', eval_keys.SUCCESS: True},
        {keys.TASK: 'complete before start task', eval_keys.POLICY_END: PolicyEnd.COMPLETE.value},
    )
    assert [episode[keys.TASK] for episode in teleop_demos(dataset=recorded)] == ['complete', 'success']


def test_teleop_demos_start_at_start_task(tmp_path):
    recorded = _record(tmp_path, {eval_keys.SUCCESS: True, ATTEMPT_START: 2500})
    [started] = teleop_demos(dataset=recorded)
    assert started.bounds(HARNESS_WORLD) == (2500, 3000)
    assert list(started[keys.GRIP].values()) == [2000.0, 3000.0]
