"""What the sound driver does with the wav its ``wav_path`` signal names."""

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pimm
from positronic.drivers import sound
from positronic.tests.testing_coutils import ManualCommandReceiver

# Captured before any test patches `wave.open`, so writing the fixture wav never goes through the spy.
_WAVE_OPEN = wave.open

SAMPLE_RATE = 44100
CHUNK_FRAMES = 64
WAV_FRAMES = 100  # one full chunk and a short one, so the file drains part-way through a run
WAV_AMPLITUDE = 8000  # int16, well clear of silence


class FakeStream:
    """Sound card that always has room for one chunk and keeps what it was handed."""

    def __init__(self):
        self.written: list[bytes] = []

    def get_write_available(self) -> int:
        return CHUNK_FRAMES

    def write(self, data: bytes) -> None:
        self.written.append(data)


class SpyWave:
    """A ``Wave_read`` that records having been closed."""

    def __init__(self, inner):
        self._inner = inner
        self.closed = False

    def getframerate(self) -> int:
        return self._inner.getframerate()

    def getsampwidth(self) -> int:
        return self._inner.getsampwidth()

    def readframes(self, n: int) -> bytes:
        return self._inner.readframes(n)

    def close(self) -> None:
        self.closed = True
        self._inner.close()


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / 'beep.wav'
    with _WAVE_OPEN(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(np.full(WAV_FRAMES, WAV_AMPLITUDE, dtype=np.int16).tobytes())
    return path


@pytest.fixture
def stream(monkeypatch) -> FakeStream:
    fake = FakeStream()
    monkeypatch.setattr(
        sound,
        'pyaudio',
        SimpleNamespace(paFloat32=object(), PyAudio=lambda: SimpleNamespace(open=lambda **kwargs: fake)),
    )
    return fake


@pytest.fixture
def opened(monkeypatch) -> list[SpyWave]:
    spies: list[SpyWave] = []

    def spy_open(path, mode):
        spies.append(SpyWave(_WAVE_OPEN(path, mode)))
        return spies[-1]

    monkeypatch.setattr(sound.wave, 'open', spy_open)
    return spies


def _play(system: sound.SoundSystem, path: Path, ticks: int) -> None:
    """Send ``path`` into the driver and run its loop ``ticks`` times."""
    commands: ManualCommandReceiver[Path] = ManualCommandReceiver()
    commands.push(path)
    system.wav_path._bind(commands)

    should_stop: ManualCommandReceiver[bool] = ManualCommandReceiver()
    should_stop.push(False)

    loop = system.run(should_stop, pimm.world.SystemClock())
    for _ in range(ticks):
        next(loop)


def _as_samples(chunk: bytes) -> np.ndarray:
    return np.frombuffer(chunk, dtype=np.float32)


def test_the_driver_plays_a_wav_named_by_a_path(wav: Path, stream: FakeStream):
    """`wave.open` opens a `str` and assumes anything else is an already-open file, so a Path reaching
    it raises `AttributeError`."""
    _play(sound.SoundSystem(), wav, ticks=1)

    assert _as_samples(stream.written[0]) == pytest.approx(np.full(CHUNK_FRAMES, WAV_AMPLITUDE / 32768.0))


def test_a_drained_wav_is_closed_and_stops_being_mixed(wav: Path, stream: FakeStream, opened: list[SpyWave]):
    # 64 + 36 frames drain the file; the third read comes back empty and ends it.
    _play(sound.SoundSystem(), wav, ticks=4)

    assert [spy.closed for spy in opened] == [True]
    assert _as_samples(stream.written[-1]) == pytest.approx(np.zeros(CHUNK_FRAMES))
