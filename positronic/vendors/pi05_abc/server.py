"""Serve a pi0.5 checkpoint fine-tuned on ABC's real data for the bottles task over positronic's session protocol.

The policy runs in the openpi environment of the author's fork, as the author's own openpi server, on this machine.
A session picks the crop of its views in the query of its URL, for example `?crop=yambox`.

Usage
  python -m positronic.vendors.pi05_abc.server bottles_16k --model.checkpoint=<checkpoint dir> --model.fork=<fork>
"""

from pathlib import Path

import configuronic as cfn
import numpy as np
from platform_client.policy_container import WARMUP_DEADLINE_S

from pimm.logging import init_logging
from positronic.offboard.server import serve
from positronic.offboard.server_utils import warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import ResizeMode, TransformImages
from positronic.vendors import abc_dit
from positronic.vendors.abc_dit import codecs as abc_codecs
from positronic.vendors.abc_dit import server as abc_server
from positronic.vendors.openpi.server import OpenpiModel, OpenpiSubprocess

# The one task of the checkpoint, and the prompt it trained under: the task folder name of the ABC-130k release.
TASK = 'put the plastic bottles in the bin'
# The checkpoint directory as the Hub repository lays it out: the author's server snapshot and the task's norm stats.
SERVER_SCRIPT = Path('inference/openpi_server.py')
NORM_STATS = Path('assets/put_the_plastic_bottles_in_the_bin/norm_stats.json')
# The fork's interpreter, and its openpi sources, where the author runs the server.
FORK_PYTHON = Path('.venv/bin/python')
FORK_OPENPI = Path('third_party/policy/openpi')
# The fork's `YamInputs` reads `observation/image`, `observation/left_wrist`, `observation/right_wrist` and
# `observation/state`; the server snapshot renames ABC's camera roles to these keys.
IMAGE_KEY_MAP = f'{abc_dit.TOP_CAMERA}=image,{abc_dit.LEFT_CAMERA}=left_wrist,{abc_dit.RIGHT_CAMERA}=right_wrist'
FLATTEN_PREFIX = 'observation/'


class AuthorServer(OpenpiSubprocess):
    """The checkpoint's own openpi server snapshot, run in the fork's interpreter on loopback.

    It loads the checkpoint and compiles the model before it listens, so it gets the deadline of a policy warm-up.
    """

    READY_TIMEOUT_S = WARMUP_DEADLINE_S

    def __init__(self, checkpoint: Path, fork: Path, config_name: str, infer_timeout: float, ws_port: int):
        # The child runs in the fork's openpi directory, where a relative path names another file.
        checkpoint, fork = checkpoint.resolve(), fork.resolve()
        super().__init__(
            checkpoint_dir=str(checkpoint),
            config_name=config_name,
            infer_timeout=infer_timeout,
            openpi_root=fork / FORK_OPENPI,
            ws_port=ws_port,
        )
        self._python = fork / FORK_PYTHON

    def _build_command(self) -> list[str]:
        checkpoint = Path(self.checkpoint_dir)
        # rules-allow: hardcoded-keys — the flags of the author's server as it parses them.
        return [
            str(self._python),
            str(checkpoint / SERVER_SCRIPT),
            '--host',
            '127.0.0.1',
            '--port',
            str(self.ws_port),
            '--config',
            self.config_name,
            '--checkpoint',
            self.checkpoint_dir,
            '--norm-stats-path',
            str(checkpoint / NORM_STATS),
            '--prompt',
            TASK,
            '--image-key-map',
            IMAGE_KEY_MAP,
            '--flatten-prefix',
            FLATTEN_PREFIX,
            '--state-key',
            f'{FLATTEN_PREFIX}state',
        ]


def _warm_observation() -> dict:
    """Black views and a zero state, in the fields of the author's server."""
    width, height = abc_codecs.VIEW_SIZE
    view = np.zeros((3, height, width), dtype=np.uint8)
    cameras = (abc_dit.TOP_CAMERA, abc_dit.LEFT_CAMERA, abc_dit.RIGHT_CAMERA)
    return {
        abc_dit.STATE: np.zeros(abc_codecs.BIMANUAL_SIZE, dtype=np.float32),
        abc_dit.IMAGES: dict.fromkeys(cameras, view),
        abc_dit.PROMPT: TASK,
    }


# `infer_timeout` bounds one inference. The author's server compiles the model before it listens.
@cfn.config(config_name='pi05_abc130k', port=8001, infer_timeout=120.0)
def pi05_abc_model(checkpoint: str, fork: str, config_name: str, port: int, infer_timeout: float) -> Model:
    """The checkpoint directory `checkpoint`, served by its own server snapshot in the fork checkout `fork`."""
    checkpoint_dir = Path(checkpoint)
    backend = AuthorServer(checkpoint_dir, Path(fork), config_name, infer_timeout, port)
    meta = {
        policy_keys.CHECKPOINT_ID: checkpoint_dir.name,
        policy_keys.TYPE: 'openpi',
        policy_keys.CONFIG_NAME: config_name,
        policy_keys.CHECKPOINT_PATH: str(checkpoint_dir),
    }
    try:
        backend.start()
        model = OpenpiModel(backend, meta)
        # The author's server swallows a failed warm-up of its own, so this one proves that the policy answers.
        warmup(model, _warm_observation())
    except Exception:
        backend.stop()
        raise
    return model


# The ABC-DiT codec, with ABC's real-data prompt of the one task.
bottles_codec = abc_codecs.multitask_200k.override(prompts={TASK: abc_codecs.ABC_130K_PROMPTS[TASK]}, prompt_prefix='')


@cfn.config(crop=abc_codecs.Crop.YAMBOX_WIDE_TOP.value)
def pipeline(crop: str) -> PolicyDeployment:
    """The views cropped by the set `crop` of `abc_codecs.CROPS`, which a session sets in its URL query. The client
    plays the first 15 actions of each 50-action chunk at 30 Hz, as the author's evaluation does, and shrinks each
    view to fit ABC's view size, or twice that size for a cropped view."""
    try:
        crop_set = abc_codecs.Crop(crop)
    except ValueError:
        raise ValueError(f'Unknown crop {crop!r}; the crops are {sorted(c.value for c in abc_codecs.Crop)}') from None
    crops = abc_codecs.CROPS[crop_set]
    codec = bottles_codec.override(crops=crops).instantiate()
    upload_size = abc_server.CROPPED_UPLOAD_SIZE if crops else abc_codecs.VIEW_SIZE
    schedule = abc_server.SCHEDULES[abc_server.ScheduleName.SYNC].client_schedule()
    return PolicyDeployment(Sequential(schedule, TransformImages(*upload_size, mode=ResizeMode.BOUND)), codec)


COMMANDS = {'bottles_16k': serve.override(model=pi05_abc_model, pipeline=pipeline)}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
