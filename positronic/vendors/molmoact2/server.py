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
from positronic.vendors.molmoact2.policy import MolmoAct2Model, warm_observation

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = 'allenai/MolmoAct2-DROID'


@cfn.config(hf_repo=DEFAULT_HF_REPO, device_map='auto', norm_tag='franka_droid', num_steps=10)
def molmoact2_model(hf_repo: str, device_map: str, norm_tag: str, num_steps: int) -> Model:
    """One pretrained MolmoAct2 checkpoint from HuggingFace, in process."""
    logger.info(f'Loading MolmoAct2 model {hf_repo} (device_map={device_map})')
    policy = MolmoAct2Model(hf_repo, device_map=device_map, norm_tag=norm_tag, num_steps=num_steps)
    warmup(policy, warm_observation())
    return policy


@cfn.config(codec=molmoact2_codecs.droid)
def pipeline(codec: Codec, fps: float = 15.0, horizon_sec: float | None = None):
    return PolicyDeployment(
        Sequential(PauseOnUnavailable(), ChunkedSchedule(fps, horizon_sec), RestrictImageSize()), codec
    )


droid = pipeline
droid_3cam = pipeline.override(codec=molmoact2_codecs.droid_3cam)


# Every pipeline is a subcommand; MolmoAct2 pins one checkpoint, so there is no separate deployment.
# The empty key is the default command, so a no-argument launch starts the server.
COMMANDS = {
    **{k: serve.override(model=molmoact2_model, pipeline=droid) for k in ('', 'serve', 'droid')},
    'droid_3cam': serve.override(model=molmoact2_model, pipeline=droid_3cam),
}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
