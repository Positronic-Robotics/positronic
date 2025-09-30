import configuronic as cfn


@cfn.config()
def linux_video(**kwargs):
    from positronic.drivers.camera.linux_video import LinuxVideo

    return LinuxVideo(**kwargs)


arducam_left = linux_video.override(
    device_path='/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684LEFT-video-index0',
    width=1920,
    height=1080,
    fps=30,
    pixel_format='MJPG',
)


arducam_right = arducam_left.override(
    device_path='/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684RIGHT-video-index0',
)


@cfn.config()
def zed(**kwargs):
    from positronic.drivers.camera.zed import SLCamera

    return SLCamera(**kwargs)


zed_m = zed.override(serial_number=17521925)
zed_2i = zed.override(serial_number=39567055)


@cfn.config()
def luxonis(**kwargs):
    from positronic.drivers.camera.luxonis import LuxonisCamera

    return LuxonisCamera(**kwargs)


@cfn.config()
def opencv(camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
    from positronic.drivers.camera.opencv import OpenCVCamera

    return OpenCVCamera(camera_id, (width, height), fps)
