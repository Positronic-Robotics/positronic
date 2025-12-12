from collections import deque
from typing import Any

from positronic.offboard.client import InferenceClient, InferenceSession
from positronic.utils import flatten_dict

from .base import Policy


class RemotePolicy(Policy):
    """
    A policy that forwards observations to a remote inference server using the
    Positronic Inference Protocol.
    """

    def __init__(self, host: str, port: int):
        self.client = InferenceClient(host, port)
        self.session: InferenceSession | None = None
        # We need to maintain the context manager to properly close it
        self._session_ctx = None
        self.action_queue = deque()

    def reset(self):
        """
        Resets the policy by starting a new session with the server.
        """
        self.action_queue.clear()
        self._close_session()
        self._session_ctx = self.client.start_session()
        self.session = self._session_ctx.__enter__()

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Forwards the observation to the remote server and returns the action.
        Uses client-side buffering if the server returns a chunk of actions.
        """
        if self.session is None:
            # If select_action is called before reset, ensure we have a session
            self.reset()

        # We know session is set by reset()
        assert self.session is not None

        if len(self.action_queue) == 0:
            result = self.session.infer(obs)
            if isinstance(result, list | tuple):
                self.action_queue.extend(result)
            else:
                self.action_queue.append(result)

        return self.action_queue.popleft()

    @property
    def meta(self) -> dict[str, Any]:
        if self.session is None:
            self.reset()

        # We know session is set by reset()
        assert self.session is not None
        return flatten_dict({'type': 'remote', 'server': self.session.metadata})

    def _close_session(self):
        if self._session_ctx:
            self._session_ctx.__exit__(None, None, None)
            self._session_ctx = None
            self.session = None

    def __del__(self):
        self._close_session()
