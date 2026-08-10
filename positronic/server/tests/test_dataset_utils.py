import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from positronic import keys
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.server.dataset_utils import (
    _MAX_PLOTTED_WIDTH,
    _collect_signal_groups,
    _unplotted_notice,
    _write_urdf_to_dir,
)


def _episode(ep_dir, widths: dict[str, int], static: dict[str, Any] | None = None) -> DiskEpisode:
    with DiskEpisodeWriter(ep_dir) as writer:
        for name, width in widths.items():
            writer.append(name, np.zeros(width, dtype=np.float32), 1000)
            writer.append(name, np.ones(width, dtype=np.float32), 2000)
        for name, value in (static or {}).items():
            writer.set_static(name, value)
    return DiskEpisode(ep_dir)


def test_narrow_signals_are_plotted(tmp_path):
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: 7, keys.GRIP: 1}))

    assert signals.plotted == {keys.JOINTS: 7, keys.GRIP: 1}
    assert signals.unplotted == {}


def test_wide_signal_is_named_instead_of_plotted(tmp_path):
    width = _MAX_PLOTTED_WIDTH + 1
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: 7, 'wide_signal': width}))

    assert signals.plotted == {keys.JOINTS: 7}
    assert signals.unplotted == {'wide_signal': width}


def test_wide_signal_still_reaches_the_3d_view(tmp_path):
    width = _MAX_PLOTTED_WIDTH + 1
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: width}))

    assert signals.numerics == [keys.JOINTS]
    assert signals.dims == {keys.JOINTS: width}


def test_every_joint_signal_the_episode_records_is_collected(tmp_path):
    widths = {'robot_state.left.q': 6, 'robot_state.right.q': 6, keys.GRIP: 1}
    static = {keys.JOINT_SIGNALS: ['robot_state.left.q', 'robot_state.right.q', 'robot_state.absent.q']}

    signals = _collect_signal_groups(_episode(tmp_path / 'ep', widths, static))

    assert sorted(signals.joints) == ['robot_state.left.q', 'robot_state.right.q']


def test_released_episodes_singular_joint_signal_still_counts(tmp_path):
    # TODO(#587): delete with the bridge in `_collect_signal_groups`.
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: 7}, {'joint_signal': keys.JOINTS}))

    assert signals.joints == [keys.JOINTS]


def test_urdf_link_and_joint_names_carry_the_namespace(tmp_path):
    urdf = """<robot name="toy">
      <link name="base"/>
      <link name="arm"/>
      <joint name="shoulder" type="revolute">
        <parent link="base"/>
        <child link="arm"/>
      </joint>
    </robot>"""

    urdf_path = _write_urdf_to_dir(urdf, {}, tmp_path, 'robot_state.left.q.')

    root = ET.fromstring(urdf_path.read_text())
    assert [el.get('name') for el in root.iter('link')] == ['robot_state.left.q.base', 'robot_state.left.q.arm']
    assert [el.get('name') for el in root.iter('joint')] == ['robot_state.left.q.shoulder']
    assert [el.get('link') for el in root.iter('parent')] == ['robot_state.left.q.base']
    assert [el.get('link') for el in root.iter('child')] == ['robot_state.left.q.arm']


def test_notice_names_every_unplotted_signal_and_its_width():
    notice = _unplotted_notice({'wide_signal': 866, 'wider_signal': 120})

    assert '`wide_signal` — 866 values' in notice
    assert '`wider_signal` — 120 values' in notice
