from typing import Any, cast

import numpy as np
import rerun.blueprint as rrb

from positronic import keys
from positronic.cfg.server import robot_replay_layout
from positronic.dataset import Time
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.dataset.signal import RECORDED_TIME
from positronic.server.dataset_utils import _build_blueprint, _collect_signal_groups

_ARM_SIGNAL_WIDTHS = {
    keys.JOINTS: 7,
    keys.EE_POSE: 7,
    keys.JOINT_VEL: 7,
    keys.GRIP: 1,
    keys.TARGET_GRIP: 1,
    keys.TARGET_JOINTS: 7,
    keys.TARGET_EE_POSE: 7,
}


def _charted_signals(item: Any) -> set[str]:
    if isinstance(item, rrb.TimeSeriesView):
        return {path.removeprefix('/signals/').removesuffix('/**') for path in cast(list[str], item.contents)}
    if isinstance(item, rrb.View):
        return set()
    return set().union(*(_charted_signals(child) for child in item.contents))


def test_the_robot_replay_layout_charts_every_signal_of_a_single_arm(tmp_path):
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        for name, width in _ARM_SIGNAL_WIDTHS.items():
            writer.append(name, np.zeros(width), Time(**{RECORDED_TIME: 1000}))
    ep = DiskEpisode(tmp_path / 'ep')
    layout = robot_replay_layout.instantiate()

    root: Any = _build_blueprint(_collect_signal_groups(ep), ep, layout).root_container
    (bottom,) = root.contents

    assert len(bottom.contents) == len(layout.bottom_row)
    assert _charted_signals(bottom) == set(_ARM_SIGNAL_WIDTHS)
