import io
from typing import Any

import msgpack
import numpy as np
import zmq

from . import Policy

###########################################################################################
# We copy the client code from gr00t N1.6 repository to reduce dependencies
# Source: gr00t/policy/server_client.py
###########################################################################################


class MsgSerializer:
    """Message serializer for ZMQ communication (N1.6 format)."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if '__ndarray_class__' in obj:
            return np.load(io.BytesIO(obj['as_npy']), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {'__ndarray_class__': True, 'as_npy': output.getvalue()}
        return obj


class PolicyClient:
    """
    Client for communicating with GR00T N1.6 PolicyServer.
    Adapted from gr00t/policy/server_client.py without BasePolicy inheritance.
    """

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
        """Kill the server."""
        self.call_endpoint('kill', requires_input=False)

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
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

        if message == b'ERROR':
            raise RuntimeError('Server error. Make sure we are running the correct policy server.')
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f'Server error: {response["error"]}')
        return response

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None):
        """
        Get action from the server.

        Args:
            observation: Dictionary of observations.
            options: Optional dictionary of options.

        Returns:
            Tuple of (action_dict, info_dict).
        """
        response = self.call_endpoint('get_action', {'observation': observation, 'options': options})
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy state."""
        return self.call_endpoint('reset', {'options': options})

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


###########################################################################################
# End of copy
###########################################################################################


class Gr00tPolicy(Policy):
    """GR00T N1.6 policy client.

    Expects observations in N1.6 nested format (from GrootInferenceObservationEncoder):
    {'video': {...}, 'state': {...}, 'language': {...}}

    Returns list of action dicts (one per time step), each with keys like:
    {'target_robot_position_translation': (3,), 'target_robot_position_quaternion': (4,), 'target_grip': (1,)}
    """

    def __init__(self, host: str, port: int, timeout_ms: int, n_action_steps: int | None = None):
        self._client = PolicyClient(host, port, timeout_ms)
        self.n_action_steps = n_action_steps

    def select_action(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        action, _info = self._client.get_action(obs)
        assert isinstance(action, dict), f'Expected dictionary, got {type(action)}'

        action = {k: v[0] for k, v in action.items()}

        lengths = {len(v) for v in action.values()}
        assert len(lengths) == 1, f'All values in action must have the same length, got {lengths}'
        time_horizon = lengths.pop()
        if self.n_action_steps is not None:
            time_horizon = min(time_horizon, self.n_action_steps)

        # Split into list of single-step actions
        # For each time step i: v[i] has shape (D,) - e.g., (3,) for translation, (1,) for grip
        return [{k: v[i] for k, v in action.items()} for i in range(time_horizon)]

    @property
    def meta(self) -> dict[str, Any]:
        # TODO: Implement metadata from server.
        return {'type': 'groot'}
