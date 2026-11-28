import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

from positronic.dataset.episode import META_UID
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.policy.dagger import DaggerDataset

# The DAgger round of the record that a training episode comes from, in the meta of the episode.
META_ROUND = 'dagger_round'
# The uid of the record that a training episode comes from, in the meta of the episode.
META_RECORD_UID = 'record_uid'


def write_training_data(rounds: Sequence[DaggerDataset], output: Path) -> None:
    """Write to the new folder ``output`` the training episodes of ``rounds[k]``, the reader's output for DAgger round
    ``k``."""
    output = output.expanduser()
    if output.exists():
        raise ValueError(f'The training data goes to a new folder, but {output} exists')
    runs = [(index, run) for index, round_data in enumerate(rounds) for run in round_data.runs]
    # `output` gets the episodes only after the last one closes, so a write that stops writes nothing to it.
    partial = output.with_name(f'.{output.name}.{uuid.uuid4().hex}')
    try:
        with LocalDatasetWriter(partial) as dataset:
            for index, run in runs:
                episode = run.episode
                meta = {META_ROUND: index, META_RECORD_UID: episode.meta[META_UID]}
                with dataset.new_episode(meta=meta) as written:
                    for name, value in episode.static.items():
                        written.set_static(name, value)
                    for name, signal in episode.signals.items():
                        for value, time in signal:
                            written.append(name, value, time)
        partial.replace(output)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
