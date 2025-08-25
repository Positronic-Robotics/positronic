from positronic.dataset.core import BaseEpisode
from positronic.dataset.transforms import Transform


class ObservationEncoder:
    def __init__(self, transforms: list[Transform], fps: int):
        self.transforms = transforms
        self.fps = fps

    def episode_to_dict(self, episode: BaseEpisode) -> dict:
        for transform in self.transforms:
            episode = transform.apply(episode)

        return episode
