import numpy as np

from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.server.dataset_utils import _MAX_PLOTTED_WIDTH, _collect_signal_groups, _unplotted_notice


def _episode(ep_dir, widths: dict[str, int]) -> DiskEpisode:
    with DiskEpisodeWriter(ep_dir) as writer:
        for name, width in widths.items():
            writer.append(name, np.zeros(width, dtype=np.float32), 1000)
            writer.append(name, np.ones(width, dtype=np.float32), 2000)
    return DiskEpisode(ep_dir)


def test_narrow_signals_are_plotted(tmp_path):
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {'robot_state.q': 7, 'grip': 1}))

    assert set(signals.numerics) == {'robot_state.q', 'grip'}
    assert signals.unplotted == {}


def test_wide_signal_is_named_instead_of_plotted(tmp_path):
    width = _MAX_PLOTTED_WIDTH + 1
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {'robot_state.q': 7, 'sim_state': width}))

    assert signals.numerics == ['robot_state.q']
    assert signals.dims == {'robot_state.q': 7}
    assert signals.unplotted == {'sim_state': width}


def test_notice_names_every_unplotted_signal_and_its_width():
    notice = _unplotted_notice({'sim_state': 866, 'contacts': 120})

    assert '`sim_state` — 866 values' in notice
    assert '`contacts` — 120 values' in notice
