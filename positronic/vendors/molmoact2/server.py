import logging
from collections.abc import Callable

import configuronic as cfn

from pimm.logging import init_logging
from positronic.offboard.server import serve
from positronic.offboard.server_utils import warmup
from positronic.offboard.spec import Model, ModelSource, PolicyDeployment
from positronic.policy import Codec, Sequential
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable
from positronic.vendors.molmoact2 import codecs as molmoact2_codecs
from positronic.vendors.molmoact2.policy import (
    BIMANUAL_YAM_STATE_DIM,
    DROID_STATE_DIM,
    MolmoAct2Model,
    warm_observation,
)

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = 'allenai/MolmoAct2-DROID'
BIMANUAL_YAM_HF_REPO = 'allenai/MolmoAct2-BimanualYAM'


class MolmoAct2Source(ModelSource):
    """Loads one pretrained MolmoAct2 checkpoint from HuggingFace into an in-process policy."""

    def __init__(
        self,
        hf_repo: str = DEFAULT_HF_REPO,
        *,
        device_map: str = 'auto',
        norm_tag: str = 'franka_droid',
        num_steps: int = 10,
        state_dim: int = DROID_STATE_DIM,
    ):
        self._hf_repo = hf_repo
        self._state_dim = state_dim
        self._device_map = device_map
        self._norm_tag = norm_tag
        self._num_steps = num_steps

    def get_models(self) -> list[str]:
        # Clients echo the advertised id onto the single-segment session route
        # (/api/v1/session/{model_id}), so it must be slash-free — derive it from the repo name.
        return [self._hf_repo.split('/')[-1]]

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Model:
        message = f'Loading MolmoAct2 model {self._hf_repo} (device_map={self._device_map})'
        logger.info(message)
        if on_progress is not None:
            on_progress(message)
        policy = MolmoAct2Model(
            self._hf_repo, device_map=self._device_map, norm_tag=self._norm_tag, num_steps=self._num_steps
        )
        warmup(policy, warm_observation(self._state_dim), on_progress)
        return policy


molmoact2_source = cfn.Config(MolmoAct2Source)


@cfn.config(codec=molmoact2_codecs.droid, source=molmoact2_source)
def pipeline(
    codec: Codec,
    source: ModelSource,
    fps: float = 15.0,
    horizon_sec: float | None = None,
    compress_images: bool = False,
):
    return PolicyDeployment(
        source,
        Sequential(PauseOnUnavailable(), ChunkedSchedule(fps, horizon_sec), RestrictImageSize()),
        codec,
        compress_images=compress_images,
    )


droid = pipeline
droid_3cam = pipeline.override(codec=molmoact2_codecs.droid_3cam)
# The checkpoint predicts 30 steps at 30 Hz; the upstream YAM example executes the first 25 of them.
yam_bimanual = pipeline.override(
    codec=molmoact2_codecs.yam_bimanual,
    source=molmoact2_source.override(
        hf_repo=BIMANUAL_YAM_HF_REPO, norm_tag='yam_dual_molmoact2', state_dim=BIMANUAL_YAM_STATE_DIM
    ),
    fps=30.0,
    horizon_sec=25 / 30,
    # Three raw frames are ~2.3 MB a request, which a lab uplink sends in most of a second.
    compress_images=True,
)


# Every pipeline is a subcommand and pins its own checkpoint, so there is no separate deployment.
# The empty key is the default command, so a no-argument launch starts the server.
COMMANDS = {
    **{k: serve.override(pipeline=droid) for k in ('', 'serve', 'droid')},
    'droid_3cam': serve.override(pipeline=droid_3cam),
    'yam_bimanual': serve.override(pipeline=yam_bimanual),
}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
