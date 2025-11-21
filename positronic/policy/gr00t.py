import io
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
import zmq
from pydantic import BaseModel

from positronic.utils import flatten_dict

from . import Policy

###########################################################################################
# We copy the code from the gr00t repository to reduce dependencies
###########################################################################################


class ModalityConfig(BaseModel):  # noqa: F821
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original
       data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if '__ModalityConfig_class__' in obj:
            obj = ModalityConfig(**json.loads(obj['as_json']))
        if '__ndarray_class__' in obj:
            obj = np.load(io.BytesIO(obj['as_npy']), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {'__ModalityConfig_class__': True, 'as_json': obj.model_dump_json()}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {'__ndarray_class__': True, 'as_npy': output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = '*', port: int = 5555, api_token: str = None):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f'tcp://{host}:{port}')
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint('ping', self._handle_ping, requires_input=False)
        self.register_endpoint('kill', self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {'status': 'ok', 'message': 'Server is running'}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get('api_token') == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f'Server is ready and listening on {addr}')
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(MsgSerializer.to_bytes({'error': 'Unauthorized: Invalid API token'}))
                    continue

                endpoint = request.get('endpoint', 'get_action')

                if endpoint not in self._endpoints:
                    raise ValueError(f'Unknown endpoint: {endpoint}')

                handler = self._endpoints[endpoint]
                result = handler.handler(request.get('data', {})) if handler.requires_input else handler.handler()
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f'Error in server: {e}')
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({'error': str(e)}))


class BaseInferenceClient:
    def __init__(self, host: str = 'localhost', port: int = 5555, timeout_ms: int = 15000, api_token: str = None):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        # Set timeout on socket before connecting
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f'tcp://{self.host}:{self.port}')

    def ping(self) -> bool:
        try:
            self.call_endpoint('ping', requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint('kill', requires_input=False)

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.

        Raises:
            RuntimeError: If the server returns an error or if a timeout occurs.
        """
        request: dict = {'endpoint': endpoint}
        if requires_input:
            request['data'] = data
        if self.api_token:
            request['api_token'] = self.api_token

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again as err:
            raise RuntimeError(
                f'Timeout after {self.timeout_ms}ms while calling endpoint "{endpoint}" at {self.host}:{self.port}'
            ) from err

        response = MsgSerializer.from_bytes(message)

        if 'error' in response:
            raise RuntimeError(f'Server error: {response["error"]}')
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint('get_action', observations)


###########################################################################################
# End of copy
###########################################################################################


class Gr00tPolicy(Policy):
    def __init__(self, host: str = 'localhost', port: int = 9000, timeout_ms: int = 15000):
        self._client = ExternalRobotInferenceClient(host, port, timeout_ms)
        self._cache = {}
        self._observation = {}
        self.server_metadata = None

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self._cache or any(not v for v in self._cache.values()):
            self._observation = flatten_dict(obs, 'observation.')
            for k in obs:
                v = obs[k]
                if isinstance(v, np.ndarray):
                    obs[k] = np.expand_dims(v, axis=0)
                    if np.issubdtype(obs[k].dtype, np.floating):
                        obs[k] = obs[k].astype(np.float64)
                else:
                    v = [v]
                    if isinstance(v[0], np.floating):
                        v = np.array(v).astype(np.float64)
                    obs[k] = v
            result = self._client.get_action(obs)
            assert isinstance(result, dict), f'Expected dictionary, got {type(result)}'

            self._cache = {k: deque(v) for k, v in result.items()}

        result = {k: v.popleft() for k, v in self._cache.items()}
        return result | self._observation

    @property
    def meta(self) -> dict[str, Any]:
        if self.server_metadata is None:
            self.server_metadata = self._client.call_endpoint('get_metadata', requires_input=False)
        result = {'type': 'groot'}
        result['server'] = self.server_metadata.copy()
        return result
