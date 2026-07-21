import asyncio
import logging
from typing import Any

import configuronic as cfn
from fastapi import WebSocket

from positronic.cfg import codecs as cfg_codecs
from positronic.offboard.vendor_server import VendorServer
from positronic.policy import Policy, PolicyWrapper
from positronic.utils.logging import init_logging
from positronic.vendors.molmoact2 import codecs as molmoact2_codecs
from positronic.vendors.molmoact2.policy import MolmoAct2Policy

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = 'allenai/MolmoAct2-DROID'


class InferenceServer(VendorServer):
    """In-process MolmoAct2 inference server for a single pretrained DROID checkpoint."""

    def __init__(
        self,
        definition: PolicyWrapper,
        hf_repo: str = DEFAULT_HF_REPO,
        *,
        device_map: str = 'auto',
        norm_tag: str = 'franka_droid',
        num_steps: int = 10,
        host: str = '0.0.0.0',
        port: int = 8000,
        recording_dir: str | None = None,
        idle_timeout_min: float | None = None,
    ):
        super().__init__(
            definition=definition, host=host, port=port, recording_dir=recording_dir, idle_timeout_min=idle_timeout_min
        )
        self.hf_repo = hf_repo
        # Clients echo the advertised id onto the single-segment session route
        # (/api/v1/session/{model_id}), so it must be slash-free — derive it from the repo name.
        self.model_id = hf_repo.split('/')[-1]
        self.device_map = device_map
        self.norm_tag = norm_tag
        self.num_steps = num_steps
        self._policy: Policy | None = None
        self.metadata = {'host': host, 'port': port, 'model_id': self.model_id, 'hf_repo': hf_repo}

    def _load_policy(self) -> Policy:
        logger.info(f'Loading MolmoAct2 model {self.hf_repo} (device_map={self.device_map})')
        return MolmoAct2Policy(
            self.hf_repo,
            device_map=self.device_map,
            norm_tag=self.norm_tag,
            num_steps=self.num_steps,
            extra_meta=self.metadata,
        )

    async def resolve_model(self, model_id: str | None, websocket: WebSocket | None) -> tuple[Any, dict]:
        if model_id is not None and model_id != self.model_id:
            raise ValueError(f'Unknown model: {model_id}. Available: {self.model_id}')
        if self._policy is None:
            self._policy = await asyncio.to_thread(self._load_policy)
        return self._policy, {'model_id': self.model_id}

    def create_policy(self, model_handle: Any) -> Policy:
        return model_handle

    async def get_models(self) -> dict:
        return {'models': [self.model_id]}

    def shutdown_model(self):
        if self._policy is not None:
            self._policy.close()
            self._policy = None


@cfn.config(
    definition=cfg_codecs.definition.override(codec=molmoact2_codecs.droid),
    hf_repo=DEFAULT_HF_REPO,
    device_map='auto',
    norm_tag='franka_droid',
    num_steps=10,
    port=8000,
    host='0.0.0.0',
    recording_dir=None,
    idle_timeout_min=None,
)
def server(
    definition: PolicyWrapper,
    hf_repo: str,
    device_map: str,
    norm_tag: str,
    num_steps: int,
    port: int,
    host: str,
    recording_dir: str | None,
    idle_timeout_min: float | None,
):
    """Starts the in-process MolmoAct2 inference server."""
    InferenceServer(
        definition=definition,
        hf_repo=hf_repo,
        device_map=device_map,
        norm_tag=norm_tag,
        num_steps=num_steps,
        host=host,
        port=port,
        recording_dir=recording_dir,
        idle_timeout_min=idle_timeout_min,
    ).serve()


if __name__ == '__main__':
    init_logging()
    cfn.cli(server)
