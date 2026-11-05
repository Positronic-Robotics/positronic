"""A stand-in for DreamAPI on loopback: it answers `/v1/get_action` from a queue, and records every request."""

import json
import threading
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

GET_ACTION = '/v1/get_action'
ROWS = 48
# The name of each status code, as the envelope of DreamAPI derives it (`shared/contracts/envelope.py`).
NAMES = {
    'DA-0000': 'OK',
    'DA-0001': 'OK_NEW_SESSION',
    'DA-2000': 'NO_CAPACITY',
    'DA-4000': 'BAD_REQUEST',
    'DA-5000': 'SERVE_FAULT',
}


def hold(observation: dict[str, Any]) -> dict[str, Any]:
    """A chunk that holds each arm at its measured state."""
    return {
        chunk: np.tile(
            observation[f'state.{slot}.joint_pos'] + observation[f'state.{slot}.gripper_pos'], (ROWS, 1)
        ).tolist()
        for slot, chunk in (('arm_l', 'q_l_chunk'), ('arm_r', 'q_r_chunk'))
    }


class FakeGateway:
    """Each `/v1/get_action` takes the next status from `statuses`, and an OK status answers `chunk(observation)`."""

    def __init__(self, chunk: Callable[[dict[str, Any]], dict[str, Any]] = hold):
        self.chunk = chunk
        self.statuses: deque[dict[str, Any]] = deque()
        self.requests: list[dict[str, Any]] = []
        self._sessions = 0
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                reply = gateway._answer(self.path, body, dict(self.headers))
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self._server.server_address[1]}'

    @property
    def calls(self) -> list[dict[str, Any]]:
        return [request for request in self.requests if request['path'] == GET_ACTION]

    @staticmethod
    def _retryable(status: dict[str, Any]) -> bool:
        """Whether a caller may send the request again, as the envelope of DreamAPI derives it."""
        if status['code'] == 'DA-5000':
            return True
        return status['code'] == 'DA-2000' and status.get('reason') != 'max_serves'

    def _answer(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        trace = f'trace-{len(self.requests) + 1}'
        self.requests.append({'path': path, 'body': body, 'headers': headers, 'trace_id': trace})
        if path != GET_ACTION:
            return {'status': {'code': 'DA-0000', 'trace_id': trace}}
        status = self.statuses.popleft() if self.statuses else {'code': 'DA-0000'}
        session = body['session_id']
        if session is None or status['code'] == 'DA-0001':
            self._sessions += 1
            session = f'session-{self._sessions}'
        envelope = {'name': NAMES[status['code']], 'retryable': self._retryable(status), 'trace_id': trace, **status}
        reply: dict[str, Any] = {'status': envelope, 'session_id': session}
        if status['code'] == 'DA-0000':
            reply['actions'] = self.chunk(body['observation'])
            reply['timing'] = {'gateway_ms': 5, 'serve_infer_ms': 3}
        return reply

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
