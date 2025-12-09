import asyncio
import logging
import traceback

import configuronic as cfn
import ormsgpack
import websockets
from websockets.server import serve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceServer:
    def __init__(self, policy, host: str, port: int):
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
            await websocket.send(ormsgpack.packb({'meta': self.policy.meta}, option=ormsgpack.OPT_SERIALIZE_NUMPY))

            # Inference Loop
            async for message in websocket:
                try:
                    obs = ormsgpack.unpackb(message)
                    action = self.policy.select_action(obs)
                    await websocket.send(ormsgpack.packb({'result': action}, option=ormsgpack.OPT_SERIALIZE_NUMPY))

                except Exception as e:
                    logger.error(f'Error processing message from {peer}: {e}')
                    logger.debug(traceback.format_exc())
                    # Send error as a string message or a special error dict
                    # For simple protocol, we might just close or send error dict
                    error_response = {'error': str(e)}
                    await websocket.send(ormsgpack.packb(error_response))

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
    cfn.cli(main)
