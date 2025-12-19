import asyncio
import logging
import traceback

import configuronic as cfn
import pos3
import websockets
from websockets.asyncio.server import serve

from positronic.offboard.serialisation import deserialise, serialise
from positronic.utils.logging import init_logging

# Configure logging
logger = logging.getLogger(__name__)


class InferenceServer:
    def __init__(self, policy, host: str, port: int):
        """
        A basic server implementation for running inference with a given policy.
        This server does not support chunking, as the Policy class itself does not,
        leading to potentially higher traffic between the client and server compared to an ideal, chunking-aware setup.
        """
        self.policy = policy
        self.host = host
        self.port = port

    async def _handler(self, websocket):
        """
        Handles the WebSocket connection.
        Protocol:
            1. Server calls policy.reset()
            2. Server sends policy.meta (packed)
            3. Server enters loop:
                - Recv obs (packed)
                - Call policy.select_action(obs)
                - Send action (packed)
        """
        peer = websocket.remote_address
        logger.info(f'Connected to {peer}')

        try:
            self.policy.reset()

            # Send Metadata
            await websocket.send(serialise({'meta': self.policy.meta}))

            # Inference Loop
            async for message in websocket:
                try:
                    obs = deserialise(message)
                    action = self.policy.select_action(obs)
                    await websocket.send(serialise({'result': action}))

                except Exception as e:
                    logger.error(f'Error processing message from {peer}: {e}')
                    logger.debug(traceback.format_exc())
                    # Send error as a string message or a special error dict
                    # For simple protocol, we might just close or send error dict
                    error_response = {'error': str(e)}
                    await websocket.send(serialise(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f'Connection closed for {peer}')
        except Exception as e:
            logger.error(f'Unexpected error for {peer}: {e}')
            logger.debug(traceback.format_exc())

    async def serve(self):
        async with serve(self._handler, self.host, self.port):
            logger.info(f'Server started on ws://{self.host}:{self.port}')
            await asyncio.get_running_loop().create_future()  # Run forever


@cfn.config(port=8000, host='0.0.0.0')
def main(policy, port: int, host: str):
    """
    Starts the inference server with the given policy.
    """
    server = InferenceServer(policy, host, port)

    # We need to run the async loop
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info('Server stopped by user')


if __name__ == '__main__':
    init_logging(logging.INFO)
    with pos3.mirror():
        cfn.cli(main)
