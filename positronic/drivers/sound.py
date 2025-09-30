import wave
from collections.abc import Iterator

import numpy as np
import pyaudio

import pimm


class SoundSystem(pimm.ControlSystem):
    def __init__(
        self,
        enable_threshold: float = 10.0,
        base_frequency: float = 220.0,
        raise_octave_each: float = 8.0,
        sample_rate: int = 44100,
        master_volume: float = 0.1,
        output_device_index: int | None = None,
    ):
        """
        This system allows you to continuously play a sound based on the level of the input.
        Args:
            enable_threshold: The threshold at which sound will be enabled.
            base_frequency: The frequency of the sound when the threshold is reached.
            raise_octave_each: Determines how much units of level will raise the frequency by one octave.
            sample_rate: Sound card sample rate.
            master_volume: The volume of the sound.
            output_device_index: The index of the output device to use.
        """
        assert sample_rate == 44100, 'Only 44100Hz sample rate is currently supported'
        assert master_volume >= 0.0 and master_volume <= 1.0, 'Master volume must be between 0 and 1'

        self.sample_rate = sample_rate
        self.enable_threshold = enable_threshold
        self.base_frequency = base_frequency
        self.raise_octave_each = raise_octave_each
        self.enable_master_volume = master_volume
        self.output_device_index = output_device_index

        self.active = True
        self.current_phase = 0.0
        self.level: pimm.SignalReceiver[float] = pimm.ControlSystemReceiver(self)
        self.wav_path: pimm.SignalReceiver[str] = pimm.ControlSystemReceiver(self)

    def _level_to_frequency(self, level: float) -> tuple[float, float]:
        if level < self.enable_threshold:
            return 0.0, self.base_frequency
        else:
            level = level - self.enable_threshold
            octave = level / self.raise_octave_each
            frequency = self.base_frequency * (2**octave)
            return self.enable_master_volume, frequency

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1024,
            output_device_index=self.output_device_index,
        )
        audio_files = {}
        file_idx = 0

        wav_path = pimm.DefaultReceiver(pimm.ValueUpdated(self.wav_path), ('', False))
        level = pimm.DefaultReceiver(self.level, 0.0)

        while not should_stop.value:
            # Load new files
            path, is_updated = wav_path.value
            if is_updated:
                print(f'Playing {path}')
                audio_files[file_idx] = wave.open(path, 'rb')
                assert audio_files[file_idx].getframerate() == 44100, 'Only 44100Hz wav files are currently supported'
                assert audio_files[file_idx].getsampwidth() == 2, 'Only 16-bit wav files are currently supported'
                file_idx += 1

            chunk_size = stream.get_write_available()
            if chunk_size == 0:
                yield pimm.Pass()
                continue

            master_volume, frequency = self._level_to_frequency(level.value)

            # Generate tone chunk
            next_chunk = self.sound_fn(chunk_size, master_volume, frequency)

            # Read audio files and mix them with the tone chunk
            finished_files = []
            for name, wave_file in audio_files.items():
                wave_chunk = wave_file.readframes(chunk_size)
                if wave_chunk is None:
                    wave_file.close()
                    finished_files.append(name)
                    yield pimm.Pass()
                    continue
                wave_chunk = np.frombuffer(wave_chunk, dtype=np.int16)

                # convert int16 to float32
                wave_chunk = wave_chunk.astype(np.float32)
                wave_chunk /= 32768.0

                next_chunk[: len(wave_chunk)] += wave_chunk

            for name in finished_files:
                del audio_files[name]

            stream.write(next_chunk.tobytes())
            yield pimm.Pass()

    def sound_fn(self, size: int, master_volume: float, frequency: float) -> np.ndarray:
        """
        This function generates a sine wave at the current frequency.

        We need to keep track of the current phase so that we don't have clicking sounds when the frequency changes.

        Args:
            size: The number of samples to generate.

        Returns:
            A numpy array containing the generated wave.
        """
        if master_volume <= 0.0:
            return np.zeros(size, dtype=np.float32)

        # Generate new wave starting from the current phase
        t = np.arange(size, dtype=np.float32)
        new_phase = self.current_phase + (np.pi * 2 * frequency / self.sample_rate * t)
        wave = np.sin(new_phase)

        # Update phase for next chunk (keep it wrapped between 0 and 2π)
        self.current_phase = np.mod(new_phase[-1] + np.pi * 2 * frequency / self.sample_rate, 2 * np.pi)

        # Apply master volume
        wave *= master_volume

        return wave
