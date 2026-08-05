import logging
from collections.abc import Callable
from typing import Any

import configuronic as cfn

from positronic.offboard.server import serve
from positronic.policy import Codec, Policy
from positronic.policy.codec import RestrictImageSize
from positronic.policy.recording import Tap
from positronic.policy.spec import ModelSource, remote
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils.logging import init_logging
from positronic.vendors.molmoact2 import codecs as molmoact2_codecs
from positronic.vendors.molmoact2.policy import MolmoAct2Policy

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = 'allenai/MolmoAct2-DROID'


class MolmoAct2Source(ModelSource):
    """Loads one pretrained MolmoAct2 checkpoint from HuggingFace into an in-process policy."""

    def __init__(
        self,
        hf_repo: str = DEFAULT_HF_REPO,
        *,
        device_map: str = 'auto',
        norm_tag: str = 'franka_droid',
        num_steps: int = 10,
    ):
        self._hf_repo = hf_repo
        self._device_map = device_map
        self._norm_tag = norm_tag
        self._num_steps = num_steps

    def get_models(self) -> list[str]:
        # Clients echo the advertised id onto the single-segment session route
        # (/api/v1/session/{model_id}), so it must be slash-free — derive it from the repo name.
        return [self._hf_repo.split('/')[-1]]

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        message = f'Loading MolmoAct2 model {self._hf_repo} (device_map={self._device_map})'
        logger.info(message)
        if on_progress is not None:
            on_progress(message)
        return MolmoAct2Policy(
            self._hf_repo, device_map=self._device_map, norm_tag=self._norm_tag, num_steps=self._num_steps
        )

    def meta(self, model_id: str) -> dict[str, Any]:
        return {'model_id': model_id, 'hf_repo': self._hf_repo}


molmoact2_source = cfn.Config(MolmoAct2Source)


@cfn.config(codec=molmoact2_codecs.droid, source=molmoact2_source)
def pipeline(codec: Codec, source: ModelSource):
    return ChunkedSchedule() | RestrictImageSize() | Tap('wire') | remote | codec | source


droid = pipeline


# Every pipeline is a subcommand; MolmoAct2 pins one checkpoint, so there is no separate deployment.
# The empty key is the default command, so a no-argument launch starts the server.
COMMANDS = {k: serve.override(pipeline=droid) for k in ('', 'serve', 'droid')}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
