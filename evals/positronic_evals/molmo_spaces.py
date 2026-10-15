"""MolmoSpaces' committed episode lists, by benchmark path below the asset packs' `benchmarks` directory.

An offline sweep builds each list. Rebuild it when the pinned MolmoSpaces revision or the asset packs change.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

# The episodes where scene geometry hides the start grasp site from the right shoulder camera. A benchmark absent
# from the list is unswept.
RIGHT_VIEW_BLOCKED_PATH = Path(__file__).with_name('right_view_blocked.json')
# The episodes whose scene cannot be built: each reset raises. The right-view sweep builds the list from the episodes
# it cannot reset.
# TODO(internal#1869): review this list as part of the MolmoSpaces upstreaming.
BROKEN_EPISODES_PATH = Path(__file__).with_name('broken_episodes.json')


def read_episode_list(path: Path) -> dict[PurePosixPath, frozenset[int]]:
    """The episodes of each benchmark that the list at `path` holds."""
    return {PurePosixPath(rel): frozenset(indices) for rel, indices in json.loads(path.read_text()).items()}
