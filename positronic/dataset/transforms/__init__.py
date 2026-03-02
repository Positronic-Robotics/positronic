"""Dataset transformation utilities."""

from positronic.utils.lazy import LazySequence, lazy_sequence

from .dataset import TransformedDataset
from .episode import EpisodeTransform, TransformedEpisode
from .signals import (
    Elementwise,
    IndexOffsets,
    Join,
    TimeOffsets,
    agg_fraction_true,
    agg_max,
    agg_mean,
    agg_percentile,
    astype,
    concat,
    diff,
    norm,
    pairwise,
    recode_rotation,
    recode_transform,
    view,
)

__all__ = [
    # Signal transforms
    'Elementwise',
    'IndexOffsets',
    'TimeOffsets',
    'Join',
    'LazySequence',
    'lazy_sequence',
    'concat',
    'astype',
    'diff',
    'norm',
    'pairwise',
    'recode_transform',
    'recode_rotation',
    'view',
    # Scalar aggregators
    'agg_fraction_true',
    'agg_max',
    'agg_mean',
    'agg_percentile',
    # Episode transforms
    'EpisodeTransform',
    'TransformedEpisode',
    # Dataset transforms
    'TransformedDataset',
]
