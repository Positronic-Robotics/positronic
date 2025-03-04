import multiprocessing

import wave
import pyaudio
import numpy as np

import ironic as ir


@ir.ironic_system(
    input_props=["level"],
    input_ports=["wav_path"],
)
class SoundSystem(ir.ControlSystem):
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
        super().__init__()
        assert sample_rate == 44100, "Only 44100Hz sample rate is currently supported"
        assert master_volume >= 0.0 and master_volume <= 1.0, "Master volume must be between 0 and 1"

        self.sample_rate = sample_rate
        self.enable_threshold = enable_threshold
        self.base_frequency = base_frequency
        self.raise_octave_each = raise_octave_each
        self.enable_master_volume = master_volume
        self.output_device_index = output_device_index

        self.manager = multiprocessing.Manager()
        self.sounds_to_play = self.manager.list()

        self.frequency = self.manager.Value("d", self.base_frequency)
        self.master_volume = self.manager.Value("d", master_volume)
        self.active = self.manager.Value("b", True)

        self.current_phase = 0.0

    async def step(self) -> ir.State:
        if self.is_bound('level'):
            level = (await self.ins.level()).data
        else:
            level = 0.0

        if level < self.enable_threshold:
            self.master_volume.value = 0.0
        else:
            self.master_volume.value = self.enable_master_volume
            level = level - self.enable_threshold
            octave = level / self.raise_octave_each
            self.frequency.value = self.base_frequency * (2 ** octave)

        return ir.State.ALIVE

    async def setup(self):
        self.thread = multiprocessing.Process(target=self._sound_thread, daemon=True)
        self.thread.start()

    async def cleanup(self):
        self.active.value = False
        self.thread.join()

    @ir.on_message('wav_path')
    async def on_wav_path(self, wav_path: ir.Message):
        print(f"Playing {wav_path.data}")
        self.sounds_to_play.append(wav_path.data)

    def _sound_thread(self):
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

        while self.active.value:
            # Load new files
            while len(self.sounds_to_play) > 0:
                name = self.sounds_to_play.pop()
                audio_files[file_idx] = wave.open(name, 'rb')
                assert audio_files[file_idx].getframerate() == 44100, "Only 44100Hz wav files are currently supported"
                assert audio_files[file_idx].getsampwidth() == 2, "Only 16-bit wav files are currently supported"
                file_idx += 1

            chunk_size = stream.get_write_available()
            if chunk_size == 0:
                continue

            # Generate tone chunk
            if self.master_volume.value > 0.0:
                next_chunk = self.sound_fn(chunk_size)
            else:
                next_chunk = np.zeros(chunk_size, dtype=np.float32)

            # Read audio files and mix them with the tone chunk
            finished_files = []
            for name, wave_file in audio_files.items():
                wave_chunk = wave_file.readframes(chunk_size)
                if wave_chunk is None:
                    wave_file.close()
                    finished_files.append(name)
                    continue
                wave_chunk = np.frombuffer(wave_chunk, dtype=np.int16)

                # convert int16 to float32
                wave_chunk = wave_chunk.astype(np.float32)
                wave_chunk /= 32768.0

                next_chunk[:len(wave_chunk)] += wave_chunk

            for name in finished_files:
                del audio_files[name]

            stream.write(next_chunk.tobytes())

    def sound_fn(self, size: int) -> np.ndarray:
        """
        This function generates a sine wave at the current frequency.

        We need to keep track of the current phase so that we don't have clicking sounds when the frequency changes.

        Args:
            size: The number of samples to generate.

        Returns:
            A numpy array containing the generated wave.
        """
        fr = self.frequency.value

        # Generate new wave starting from the current phase
        t = np.arange(size, dtype=np.float32)
        new_phase = self.current_phase + (np.pi * 2 * fr / self.sample_rate * t)
        wave = np.sin(new_phase)

        # Update phase for next chunk (keep it wrapped between 0 and 2π)
        self.current_phase = (new_phase[-1] + np.pi * 2 * fr / self.sample_rate) % (2 * np.pi)

        # Apply master volume
        wave *= self.master_volume.value

        return wave
