import configuronic as cfgc
from pimm.drivers.camera.linux_video import LinuxVideo


arducam_left = cfgc.Config(
    LinuxVideo,
    device_path="/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684LEFT-video-index0",
    width=1920,
    height=1080,
    fps=30,
    pixel_format="MJPG",
)

arducam_right = arducam_left.override(
    device_path="/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684RIGHT-video-index0",
)
