import logging

import configuronic as cfn

from pimm.logging import init_logging
from positronic.offboard.server import serve
from positronic.offboard.server_utils import warmup
from positronic.offboard.spec import Model, PolicyDeployment
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


@cfn.config(
    hf_repo=DEFAULT_HF_REPO, device_map='auto', norm_tag='franka_droid', num_steps=10, state_dim=DROID_STATE_DIM
)
def molmoact2_model(hf_repo: str, device_map: str, norm_tag: str, num_steps: int, state_dim: int) -> Model:
    """One pretrained MolmoAct2 checkpoint from HuggingFace, in process."""
    logger.info(f'Loading MolmoAct2 model {hf_repo} (device_map={device_map})')
    policy = MolmoAct2Model(hf_repo, device_map=device_map, norm_tag=norm_tag, num_steps=num_steps)
    warmup(policy, warm_observation(state_dim))
    return policy


@cfn.config(codec=molmoact2_codecs.droid)
def pipeline(codec: Codec, fps: float = 15.0, horizon_sec: float | None = None, compress_images: bool = False):
    return PolicyDeployment(
        Sequential(PauseOnUnavailable(), ChunkedSchedule(fps, horizon_sec), RestrictImageSize()),
        codec,
        compress_images=compress_images,
    )


droid = pipeline
droid_3cam = pipeline.override(codec=molmoact2_codecs.droid_3cam)
# The checkpoint predicts 30 steps at 30 Hz; the upstream YAM example executes the first 25 of them.
yam_bimanual = pipeline.override(
    codec=molmoact2_codecs.yam_bimanual,
    fps=30.0,
    horizon_sec=25 / 30,
    # Three raw frames are ~2.3 MB a request, which a lab uplink sends in most of a second.
    compress_images=True,
)
yam_bimanual_model = molmoact2_model.override(
    hf_repo=BIMANUAL_YAM_HF_REPO, norm_tag='yam_dual_molmoact2', state_dim=BIMANUAL_YAM_STATE_DIM
)


# Every pipeline is a subcommand and pins its own checkpoint, so there is no separate deployment.
# The empty key is the default command, so a no-argument launch starts the server.
COMMANDS = {
    **{k: serve.override(model=molmoact2_model, pipeline=droid) for k in ('', 'serve', 'droid')},
    'droid_3cam': serve.override(model=molmoact2_model, pipeline=droid_3cam),
    'yam_bimanual': serve.override(model=yam_bimanual_model, pipeline=yam_bimanual),
}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
