import asyncio
import logging
import traceback
from collections.abc import Callable
from typing import Any

import configuronic as cfn
import pos3
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from positronic.offboard.serialisation import deserialise, serialise
from positronic.utils.logging import init_logging

# Configure logging
logger = logging.getLogger(__name__)


class InferenceServer:
    def __init__(self, policy_registry: dict[str, Callable[[], Any]], host: str, port: int):
        """
        A basic server implementation for running inference with multiple policies.
        Serve policies based on the request path.
        """
        self.policy_registry = policy_registry
        self.host = host
        self.port = port
        self.app = FastAPI()

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
            logger.info(f'Using default policy: {self.default_key}')
        elif model_id in self.policy_registry:
            policy_factory = self.policy_registry[model_id]
        else:
            logger.error(f'Policy not found: {model_id}')
            # Reject connection with specific code
            await websocket.close(code=1008, reason='Policy not found')
            return

        try:
            policy = policy_factory()
            policy.reset()

            # Send Metadata
            await websocket.send_bytes(serialise({'meta': policy.meta}))

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
        server = uvicorn.Server(config)
        return server.serve()


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
