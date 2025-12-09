from typing import Any

from positronic.inference.client import InferenceClient, InferenceSession
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

    def reset(self):
        """
        Resets the policy by starting a new session with the server.
        """
        self._close_session()
        self._session_ctx = self.client.start_session()
        self.session = self._session_ctx.__enter__()

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Forwards the observation to the remote server and returns the action.
        """
        if self.session is None:
            # If select_action is called before reset, ensure we have a session
            self.reset()

        # We know session is set by reset()
        assert self.session is not None
        return self.session.infer(obs)

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
