from typing import Optional

import numpy as np

import ironic as ir


def _sound_system(force_feedback_volume: Optional[float] = None,
                  start_recording_wav_path: Optional[str] = None,
                  stop_recording_wav_path: Optional[str] = None):
    from drivers.sound import SoundSystem

    sound_system = SoundSystem(master_volume=force_feedback_volume or 0)
    components = [sound_system]
    inputs = {'force': None, 'start_recording': None, 'stop_recording': None}

    if force_feedback_volume is not None:
        map_force = ir.utils.MapPropertyCS(np.linalg.norm)
        sound_system.bind(level=map_force.outs.output)
        inputs['force'] = (map_force, 'input')
        components.append(map_force)

    if start_recording_wav_path is not None:
        map_start_recording = ir.utils.MapPortCS(lambda _: start_recording_wav_path)
        sound_system.bind(wav_path=map_start_recording.outs.output)
        inputs['start_recording'] = (map_start_recording, 'input')
        components.append(map_start_recording)

    if stop_recording_wav_path is not None:
        map_stop_recording = ir.utils.MapPortCS(lambda _: stop_recording_wav_path)
        sound_system.bind(wav_path=map_stop_recording.outs.output)
        inputs['stop_recording'] = (map_stop_recording, 'input')
        components.append(map_stop_recording)

    return ir.compose(*components, inputs=inputs, outputs=sound_system.output_mappings)


full = ir.Config(_sound_system,
                 force_feedback_volume=0.1,
                 start_recording_wav_path="assets/sounds/recording-has-started.wav",
                 stop_recording_wav_path="assets/sounds/recording-has-stopped.wav")

start_stop = full.override(force_feedback_volume=0)
