from ..dataset import Dataset
from ..episode import Episode
from ..signal import SignalMeta
from .episode import EpisodeTransform, TransformedEpisode


class TransformedDataset(Dataset):
    """Transform a dataset into a new view of the dataset."""

    def __init__(self, dataset: Dataset, *transforms: EpisodeTransform, pass_through: bool | list[str] = False):
        self._dataset = dataset
        self._transforms = transforms
        self._pass_through = pass_through
        if not isinstance(pass_through, bool):
            self._pass_through = tuple(pass_through)
        self._meta = None

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        episode = self._dataset[index]
        return TransformedEpisode(episode, *self._transforms, pass_through=self._pass_through)

    @property
    def signals_meta(self) -> dict[str, SignalMeta]:
        if self._meta is None:
            ep = self[0]
            self._meta = {name: ep[name].meta for name in ep.signals.keys()}
        return self._meta
