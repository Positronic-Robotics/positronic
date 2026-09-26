"""A server that returns empty commands after a configurable delay.

The session parameter ``?delay_sec=120`` holds each inference open for two minutes.
"""

import time
from typing import Any

import configuronic as cfn

from pimm.logging import init_logging
from positronic.offboard import keys as offboard_keys
from positronic.offboard.server import serve
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy.base import Obs
from positronic.policy.codec import Codec
from positronic.policy.layers import ChunkedSchedule


class StubModel(Model):
    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        return [{}]

    def meta(self) -> dict[str, Any]:
        return {'model_name': 'stub', offboard_keys.CHECKPOINT_ID: 'stub'}


class Delay(Codec):
    """Delay the call without changing its inputs or outputs."""

    def __init__(self, delay_sec: float):
        self._delay_sec = delay_sec

    def encode(self, data):
        time.sleep(self._delay_sec)
        return data

    def decode(self, data):
        return data


@cfn.config(delay_sec=0.0)
def pipeline(delay_sec: float) -> PolicyDeployment:
    return PolicyDeployment(ChunkedSchedule(fps=15), Delay(delay_sec))


if __name__ == '__main__':
    init_logging()
    cfn.cli(serve.override(model=cfn.Config(StubModel), pipeline=pipeline))
