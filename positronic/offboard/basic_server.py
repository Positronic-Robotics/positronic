import asyncio
import logging
import traceback
from collections.abc import Callable
from typing import Any

import configuronic as cfn
import pos3
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialise, serialise

# Configure logging
logger = logging.getLogger(__name__)


class InferenceServer:
    def __init__(self, policy_registry: dict[str, Callable[[], Any]] | Any, host: str, port: int):
        """
        A basic server implementation for running inference with multiple policies.
        Serve policies based on the request path.
        """
        if isinstance(policy_registry, dict):
            self.policy_registry = policy_registry
        else:
            self.policy_registry = {'default': lambda: policy_registry}

        if not self.policy_registry:
            raise ValueError('policy_registry must contain at least one policy')
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.server: uvicorn.Server | None = None

        self.default_key = next(iter(self.policy_registry))

        # Register routes
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.websocket_endpoint)
        self.app.websocket('/api/v1/session/{model_id}')(self.websocket_endpoint)

    async def get_models(self):
        return {'models': list(self.policy_registry.keys())}

    async def websocket_endpoint(self, websocket: WebSocket, model_id: str | None = None):
        await websocket.accept()
        logger.info(f'Connected to {websocket.client} requesting {model_id or "default"}')

        # Resolve policy
        if not model_id:
            policy_factory = self.policy_registry[self.default_key]
        elif model_id in self.policy_registry:
            policy_factory = self.policy_registry[model_id]
        else:
            logger.error(f'Policy not found: {model_id}')
            # Send error status
            await websocket.send_bytes(serialise({'status': 'error', 'error': f'Policy not found: {model_id}'}))
            await websocket.close(code=1008, reason='Policy not found')
            return

        try:
            # Send loading status before blocking operation
            await websocket.send_bytes(serialise({'status': 'loading', 'message': 'Loading policy...'}))

            policy = policy_factory()
            policy.reset()

            # Send ready with metadata
            await websocket.send_bytes(serialise({'status': 'ready', 'meta': policy.meta}))

            # Inference Loop
            async for message in websocket.iter_bytes():
                try:
                    obs = deserialise(message)
                    action = policy.select_action(obs)
                    await websocket.send_bytes(serialise({'result': action}))

                except Exception as e:
                    logger.error(f'Error processing message: {e}')
                    logger.debug(traceback.format_exc())
                    error_response = {'error': str(e)}
                    await websocket.send_bytes(serialise(error_response))

        except (WebSocketDisconnect, Exception) as e:
            logger.info(f'Connection closed: {e}')
            logger.debug(traceback.format_exc())

    def serve(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='info')
        self.server = uvicorn.Server(config)
        return self.server.serve()

    def shutdown(self):
        if self.server is not None:
            self.server.should_exit = True


@cfn.config(port=8000, host='0.0.0.0')
def main(policy, port: int, host: str):
    """
    Starts the inference server with the given policy.
    """
    # Wrap single policy in a registry for backward compatibility
    registry = {'default': lambda: policy}
    server = InferenceServer(registry, host, port)

    # Run the server
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info('Server stopped by user')


if __name__ == '__main__':
    init_logging(logging.INFO)
    with pos3.mirror():
        cfn.cli(main)
