import configuronic as cfn

from positronic import keys


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
    device_path='/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684RIGHT-video-index0'
)


@cfn.config()
def zed(**kwargs):
    from positronic.drivers.camera.zed import SLCamera

    return SLCamera(**kwargs)


zed_m = zed.override(serial_number=17521925)
zed_2i = zed.override(serial_number=39567055)
zed_2i_second = zed.override(serial_number=13785037)

_DROID_STREAM = {'view': 'left', 'resolution': 'hd720', 'fps': 30, 'image_enhancement': False}

droid = {keys.WRIST_IMAGE: zed_m.override(**_DROID_STREAM), keys.EXTERIOR_IMAGE: zed_2i.override(**_DROID_STREAM)}

droid_3cam = {**droid, keys.EXTERIOR_IMAGE_2: zed_2i_second.override(**_DROID_STREAM)}

# YAM station (brunello): ZED X overhead + two ZED X One wrist cameras on the ZED Link Duo.
zed_x_top = zed.override(serial_number=48953814)
zed_x_one_left = zed.override(serial_number=309745677, mono=True)
zed_x_one_right = zed.override(serial_number=303714482, mono=True)


@cfn.config()
def luxonis(**kwargs):
    from positronic.drivers.camera.luxonis import LuxonisCamera

    return LuxonisCamera(**kwargs)


@cfn.config()
def opencv(camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
    from positronic.drivers.camera.opencv import OpenCVCamera

    return OpenCVCamera(camera_id, (width, height), fps)
