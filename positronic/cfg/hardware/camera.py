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


# rules-allow: primitive-type — a device path is the `str` `linuxpy.Device` takes and `LinuxVideo` hands it
# The four RealSense D405 of the Trossen station. A D405 enumerates as a plain UVC device with six video
# nodes, of which `-video-index4` carries the colour stream. The serial in the link is the USB one, which
# is not the serial the RealSense SDK reports for the same camera.
_D405_DEVICE_PATH_PREFIX = (
    '/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405_'
)

d405_wrist_left = linux_video.override(
    device_path=f'{_D405_DEVICE_PATH_PREFIX}251323070021-video-index4',
    width=640,
    height=480,
    fps=30,
    pixel_format='YUYV',
)

d405_wrist_right = d405_wrist_left.override(device_path=f'{_D405_DEVICE_PATH_PREFIX}251323070565-video-index4')

d405_scene_top = d405_wrist_left.override(device_path=f'{_D405_DEVICE_PATH_PREFIX}260323072626-video-index4')

d405_scene_bottom = d405_wrist_left.override(device_path=f'{_D405_DEVICE_PATH_PREFIX}260323072970-video-index4')


@cfn.config()
def zed(**kwargs):
    from positronic.drivers.camera.zed import SLCamera

    return SLCamera(**kwargs)


zed_m = zed.override(serial_number=17521925)
zed_2i = zed.override(serial_number=39567055)
zed_2i_second = zed.override(serial_number=39058547)

# The station's two sideviews, by side, read off the rig (Positronic-Robotics/internal#1131).
sideview_left = zed_2i
sideview_right = zed_2i_second

_DROID_STREAM = {'view': 'left', 'resolution': 'hd720', 'fps': 30, 'image_enhancement': False}

droid = {
    keys.WRIST_IMAGE: zed_m.override(**_DROID_STREAM),
    keys.EXTERIOR_IMAGE: sideview_left.override(**_DROID_STREAM),
}

droid_3cam = {**droid, keys.EXTERIOR_IMAGE_2: sideview_right.override(**_DROID_STREAM)}

# `droid_left` holds `sideview_left` under `exterior`; `droid_right` holds `sideview_right` there.
# FOOTGUN: `exterior` and `exterior_2` hold opposite sideviews in the two dicts.
droid_left = droid_3cam  # the unsided three-camera set already binds the left sideview as `exterior`
droid_right = {
    **droid_3cam,
    keys.EXTERIOR_IMAGE: sideview_right.override(**_DROID_STREAM),
    keys.EXTERIOR_IMAGE_2: sideview_left.override(**_DROID_STREAM),
}

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
