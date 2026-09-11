from positronic.drivers.camera import zed
from positronic.drivers.camera.zed import CAMERA_STATE_SETTINGS, read_camera_state

# The SDK namespace the driver bound at import: the vendor's when installed, the conftest's stand-in otherwise.
sl = zed.sl


class FakeCamera:
    """Answers each setting with the value it holds, and refuses the ones it does not."""

    def __init__(self, values: dict):
        self._values = values

    def get_camera_settings(self, setting) -> tuple:
        if setting in self._values:
            return sl.ERROR_CODE.SUCCESS, self._values[setting]
        return sl.ERROR_CODE.FAILURE, -1


def test_read_camera_state_reports_every_setting_the_camera_answers():
    camera = FakeCamera({
        sl.VIDEO_SETTINGS.EXPOSURE: 45,
        sl.VIDEO_SETTINGS.GAIN: 12,
        sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE: 4700,
        sl.VIDEO_SETTINGS.AEC_AGC: 1,
        sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO: 1,
    })

    assert read_camera_state(camera) == {
        'exposure': 45,
        'gain': 12,
        'white_balance_temperature': 4700,
        'auto_exposure': 1,
        'auto_white_balance': 1,
    }


def test_read_camera_state_leaves_out_a_setting_the_camera_refuses():
    camera = FakeCamera({sl.VIDEO_SETTINGS.EXPOSURE: 45})

    assert read_camera_state(camera) == {'exposure': 45}


def test_every_recorded_setting_names_a_video_setting():
    assert all(hasattr(sl.VIDEO_SETTINGS, sdk_name) for sdk_name in CAMERA_STATE_SETTINGS.values())
