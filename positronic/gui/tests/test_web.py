import queue
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import numpy as np
import pytest
from eval_vocabulary.outcome import OUTCOME, Outcome
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import pimm
from pimm.time import EMITTED_WORLD
from pimm.world import SystemClock
from positronic.dataset.episode import META_UID
from positronic.dataset.local_dataset import LocalDatasetWriter, load_dataset
from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.gui.station import Ending, Episode, Phase, RunView, Station, Unrecorded, terminal_payload
from positronic.gui.web import STATIC_DIR, CameraFeed, CameraView, InstructionBody, StationConsole, Status, VerdictBody
from positronic.policy import Policy
from positronic.policy import keys as policy_keys
from positronic.policy.harness import EpisodeEnd
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

CONFIGURED = 'put the cup in the tote'
OVERRIDE = 'put the red cup in the grey tote'
CAMERA = 'image.exterior_2'


def _trial() -> Task:
    return Task(instruction_source=CONFIGURED, timeout_sec=None)


def _frame(i: int) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, : 20 * i % 640] = 200
    return frame


class _Policy(Policy):
    def meta(self):
        return {policy_keys.TYPE: 'remote'}

    def run(self, runtime):
        yield
        return eval_keys.PolicyEnd.COMPLETE


class _Console:
    """A station page served in-process, with queued tasks and emitted verdicts collected for inspection."""

    def __init__(self):
        self.tasks: list[Task] = []
        self.done = RecordingEmitter[dict[str, Any]]()
        self.feed = CameraFeed()
        self.should_stop = ManualCommandReceiver[bool]()
        self.should_stop.push(False)
        self.recorded: list[tuple[str, Outcome]] = []
        self.station = Station(_trial, lambda uid, outcome: self.recorded.append((uid, outcome)))
        console = StationConsole(_trial, _Policy(), None, host='127.0.0.1', port=0)
        console.done._bind(self.done, clock=SystemClock())
        app = console.build_app(self.station, {CAMERA: self.feed}, self.tasks.append, SystemClock(), self.should_stop)
        self.client = TestClient(app)

    def post(self, path: str, body: VerdictBody | InstructionBody | None = None) -> Status:
        response = self.client.post(path, json=None if body is None else body.model_dump(mode='json'))
        response.raise_for_status()
        return Status.model_validate(response.json())


@pytest.fixture
def console() -> Iterator[_Console]:
    console = _Console()
    yield console
    console.should_stop.push(True)
    console.feed.stream.close()


def test_the_page_and_every_asset_it_links_are_served(console):
    page = console.client.get('/')
    assert 'Station console' in page.text
    assets = [link for link in re.findall(r'(?:href|src)="([^"]+)"', page.text) if not link.startswith('data:')]
    assert sorted(assets) == ['station.css', 'station.js']
    for asset in assets:
        assert console.client.get(f'/{asset}').status_code == 200


def test_the_status_is_the_json_the_page_reads(console):
    console.feed.push(_frame(1), SystemClock().now())
    status = console.client.get('/status').json()
    status['run'].pop('now')
    assert isinstance(status['run'].pop('generation'), int)
    assert status == {
        'run': {'phase': 'ready', 'configured': CONFIGURED, 'override': None, 'override_since': None, 'episodes': []},
        'cameras': [{'name': CAMERA, 'label': 'exterior 2', 'live': True, 'fps': 0.0, 'width': 640, 'height': 360}],
        'policy': 'remote',
        'host': '127.0.0.1',
    }


def test_start_hands_the_trial_to_the_control_loop_once(console):
    console.post('/instruction', InstructionBody(override=OVERRIDE))
    assert console.post('/episode/start').run.phase is Phase.RUNNING
    [task] = console.tasks
    assert isinstance(task, Task) and task.instruction == OVERRIDE

    again = console.client.post('/episode/start')
    assert again.status_code == 409
    assert 'already running' in again.json()['detail']
    assert len(console.tasks) == 1


def test_each_verdict_emits_done_without_waiting_for_the_control_loop(console):
    console.post('/episode/start')
    assert console.post('/episode/end', VerdictBody(verdict=Outcome.FAIL)).run.phase is Phase.ENDING
    assert len(console.tasks) == 1
    [(_, payload)] = console.done.emitted
    assert payload == terminal_payload(Outcome.FAIL)
    assert console.post('/episode/end', VerdictBody(verdict=Outcome.DISCARDED)).run.phase is Phase.ENDING
    assert [payload for _, payload in console.done.emitted] == [
        terminal_payload(Outcome.FAIL),
        terminal_payload(Outcome.DISCARDED),
    ]


def test_the_page_offers_only_the_verdicts(console):
    console.post('/episode/start')
    assert console.client.post('/episode/end', json={'verdict': Outcome.OUT_OF_TIME.value}).status_code == 422
    assert len(console.tasks) == 1


POLICY_END = EpisodeEnd({eval_keys.ENDED_BY: eval_keys.ENDED_BY_POLICY, eval_keys.TERMINATED: True}, 'recording-1')


def test_after_a_policy_end_the_verdict_closes_the_episode_and_emits_no_done(console):
    console.post('/episode/start')
    console.station.close(POLICY_END, now=1.0)
    assert Status.model_validate(console.client.get('/status').json()).run.phase is Phase.VERDICT_PENDING
    status = console.post('/episode/end', VerdictBody(verdict=Outcome.SUCCESS))
    assert status.run.phase is Phase.READY
    assert status.run.episodes[0].outcome is Outcome.SUCCESS
    assert console.recorded == [('recording-1', Outcome.SUCCESS)]
    assert console.done.emitted == []


def test_the_instruction_is_refused_while_an_episode_runs(console):
    console.post('/episode/start')
    response = console.client.post('/instruction', json=InstructionBody(override=OVERRIDE).model_dump())
    assert response.status_code == 409
    assert console.station.view(now=0.0).override is None


def test_end_run_is_refused_while_an_episode_runs_and_ends_the_run_after_it(console):
    console.post('/episode/start')
    refused = console.client.post('/run/end')
    assert refused.status_code == 409
    assert refused.json()['detail'] == 'finish the episode first, then end the run'
    console.station.close(EpisodeEnd({eval_keys.TERMINATED: False}, None), now=1.0)
    assert console.post('/run/end').run.phase is Phase.RUN_ENDED


def test_a_post_from_another_site_is_refused(console):
    refused = console.client.post('/episode/start', headers={'Origin': 'http://example.com'})
    assert refused.status_code == 403
    assert console.tasks == []
    accepted = console.client.post('/episode/start', headers={'Origin': 'http://testserver'})
    assert accepted.status_code == 200


PAGE_ORIGIN = {'Origin': 'http://testserver'}


def test_a_tile_streams_its_codec_its_init_segment_and_then_fragments_to_the_page(console):
    for i in range(20):
        console.feed.push(_frame(i), SystemClock().now())
    with console.client.websocket_connect(f'/video/{CAMERA}', headers=PAGE_ORIGIN) as socket:
        assert socket.receive_text().startswith('avc1.')
        assert socket.receive_bytes() == console.feed.stream.init_segment
        for i in range(20, 40):
            console.feed.push(_frame(i), SystemClock().now())
        assert socket.receive_bytes()[4:8] == b'moof'
        console.should_stop.push(True)


def test_a_tile_is_refused_to_a_page_from_another_site(console):
    for i in range(20):
        console.feed.push(_frame(i), SystemClock().now())
    with pytest.raises(WebSocketDisconnect):
        with console.client.websocket_connect(f'/video/{CAMERA}', headers={'Origin': 'http://example.com'}) as socket:
            socket.receive_text()


def test_a_tile_for_an_unknown_camera_is_refused(console):
    with pytest.raises(WebSocketDisconnect):
        with console.client.websocket_connect('/video/image.nowhere') as socket:
            socket.receive_text()


def test_a_camera_reads_live_with_its_rate_until_its_frames_stop():
    feed = CameraFeed()
    for i in range(11):
        feed.push(_frame(i), now=100.0 + i * 0.1)
    live = feed.view(CAMERA, now=101.05)
    assert live.live and live.fps == pytest.approx(10.0)
    stale = feed.view(CAMERA, now=103.0)
    assert not stale.live and stale.fps == 0.0
    assert (stale.width, stale.height) == (640, 360)
    feed.stream.close()


class _Frame:
    def __init__(self, array: np.ndarray):
        self.array = array


def _console_with_camera() -> tuple[StationConsole, ManualCommandReceiver, CameraFeed]:
    console = StationConsole(_trial, _Policy(), None, host='127.0.0.1', port=0)
    camera = ManualCommandReceiver()
    console.cameras[CAMERA]._bind(camera)
    return console, camera, CameraFeed()


def test_a_camera_with_no_data_keeps_its_last_tile():
    console, camera, feed = _console_with_camera()
    camera.push(_Frame(_frame(1)))
    console._push_frames({CAMERA: feed}, SystemClock())
    camera.push(pimm.SignalError('camera lost'))
    console._push_frames({CAMERA: feed}, SystemClock())

    assert feed.view(CAMERA, now=0.0).width == 640
    feed.stream.close()


def test_a_camera_that_gives_data_again_updates_its_tile():
    console, camera, feed = _console_with_camera()
    camera.push(pimm.SignalError('camera lost'))
    console._push_frames({CAMERA: feed}, SystemClock())
    camera.push(_Frame(_frame(1)))
    console._push_frames({CAMERA: feed}, SystemClock())

    assert feed.view(CAMERA, now=0.0).width == 640
    feed.stream.close()


class _Answered(pimm.calls.Answer[EpisodeEnd]):
    """An episode answer that the harness gave."""

    def __init__(self, end: EpisodeEnd):
        self._end = end

    def done(self) -> bool:
        return True

    def result(self) -> EpisodeEnd:
        return self._end


def _recorded_episode(path: Path) -> str:
    """Record one episode into the dataset at ``path``, and return its uid."""
    with LocalDatasetWriter(path) as dataset, dataset.new_episode() as episode:
        episode.append('signal', 1, pimm.Time(**{EMITTED_WORLD: 0}))
        episode.set_static(eval_keys.TERMINATED, False)
        return episode.meta[META_UID]


def test_an_episode_that_runs_out_of_time_gets_its_outcome_in_the_edit_log(tmp_path):
    uid = _recorded_episode(tmp_path)
    console = StationConsole(_trial, _Policy(), tmp_path, host='127.0.0.1', port=0)
    station = Station(_trial, console._record_outcome)
    station.start(now=0.0)
    console._episode = _Answered(EpisodeEnd({eval_keys.TERMINATED: False}, uid))

    console._drive_episode(station, queue.SimpleQueue(), SystemClock())

    [episode] = load_dataset(tmp_path)
    assert episode[OUTCOME] == Outcome.OUT_OF_TIME
    assert station.view(now=0.0).episodes[0].outcome is Outcome.OUT_OF_TIME


def test_a_verdict_after_a_policy_end_goes_into_the_edit_log(tmp_path):
    uid = _recorded_episode(tmp_path)
    console = StationConsole(_trial, _Policy(), tmp_path, host='127.0.0.1', port=0)
    station = Station(_trial, console._record_outcome)
    station.start(now=0.0)
    console._episode = _Answered(EpisodeEnd(POLICY_END.terminal, uid))
    console._drive_episode(station, queue.SimpleQueue(), SystemClock())
    assert OUTCOME not in load_dataset(tmp_path)[0]

    station.end(Outcome.DISCARDED)

    assert load_dataset(tmp_path)[0][OUTCOME] == Outcome.DISCARDED


class _Failed(pimm.calls.Answer[EpisodeEnd]):
    """An episode answer that the harness failed with ``error``."""

    def __init__(self, error: Exception):
        self._error = error

    def done(self) -> bool:
        return True

    def result(self) -> EpisodeEnd:
        raise self._error


def test_an_arm_error_that_fails_an_episode_reaches_the_page_and_is_not_skipped_as_a_frame_is():
    console = StationConsole(_trial, _Policy(), None, host='127.0.0.1', port=0)
    station = Station(_trial, console._record_outcome)
    station.start(now=0.0)
    console._episode = _Failed(pimm.SignalError('Release the emergency stop button'))

    console._drive_episode(station, queue.SimpleQueue(), SystemClock())

    [episode] = station.view(now=0.0).episodes
    assert episode.outcome is Unrecorded.ERROR
    assert episode.error is not None and episode.error.endswith('Release the emergency stop button')


SCRIPT = (STATIC_DIR / 'station.js').read_text()


def _script_part(pattern: str, flags: re.RegexFlag) -> str:
    match = re.search(pattern, SCRIPT, flags)
    assert match is not None, f'the page script has no {pattern}'
    return match.group(1)


def _script_object(name: str) -> str:
    """The body of ``const <name> = {...};`` in the page script."""
    return _script_part(rf'const {name} = {{(.*?)}};', re.DOTALL)


def _members(annotation) -> Iterator[Any]:
    """The values of a union of ``Literal`` types and enums."""
    for arg in get_args(annotation):
        yield from get_args(arg) if get_origin(arg) is Literal else arg


def test_the_page_names_the_phases_and_outcomes_the_server_sends():
    assert set(re.findall(r"'([^']*)'", _script_object('PHASE'))) == {phase.value for phase in Phase}
    assert set(re.findall(r"'([^']*)'", _script_object('OUTCOME'))) == {outcome.value for outcome in _members(Ending)}


def test_the_page_sends_the_request_fields_the_server_reads():
    sent = set(re.findall(r'\(\{ (\w+):', _script_object('REQUEST')))
    assert sent == set(InstructionBody.model_fields) | set(VerdictBody.model_fields)


def test_the_page_reads_every_status_field_the_server_sends():
    reader = _script_part(r'^function readStatus\(answer\) \{(.*?)^\}$', re.MULTILINE | re.DOTALL)
    read = set(re.findall(r'\b(?:answer|run|e|c)\.(\w+)', reader))
    assert read == {field for model in (Status, RunView, Episode, CameraView) for field in model.model_fields}


def test_the_page_calls_the_routes_the_server_serves(console):
    served = {route.path for route in console.client.app.routes}
    routes = dict(re.findall(r"(\w+): '([^']*)'", _script_object('ROUTES')))
    video = routes.pop('video')
    assert set(routes.values()) <= served
    assert f'{video}/{{name}}' in served


def test_a_status_taken_before_a_change_is_older_than_the_answer_to_the_change(console):
    before = Status.model_validate(console.client.get('/status').json())
    after = console.post('/instruction', InstructionBody(override=OVERRIDE))
    assert after.run.generation > before.run.generation
