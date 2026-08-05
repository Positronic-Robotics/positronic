import numpy as np

from positronic import keys
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.server.dataset_utils import _MAX_PLOTTED_WIDTH, _collect_signal_groups, _unplotted_notice


def _episode(ep_dir, widths: dict[str, int]) -> DiskEpisode:
    with DiskEpisodeWriter(ep_dir) as writer:
        for name, width in widths.items():
            writer.append(name, np.zeros(width, dtype=np.float32), 1000)
            writer.append(name, np.ones(width, dtype=np.float32), 2000)
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


def test_notice_names_every_unplotted_signal_and_its_width():
    notice = _unplotted_notice({'wide_signal': 866, 'wider_signal': 120})

    assert '`wide_signal` — 866 values' in notice
    assert '`wider_signal` — 120 values' in notice
