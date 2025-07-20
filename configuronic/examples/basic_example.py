import configuronic as cfgc


def basic_camera(camera_id: int, fps):
    cam = some_library.Camera(camera_id)
    camera.set_fps(fps)
    return cam


basic_camera_0 = cfgc.Config(camera_id=0, fps=30)
basic_camera_1 = basic_camera_0.override(camera_id=1)


@cfgc.config(usb_path='/dev/v4l/by-id/usb-video-index0')
def v4l_camera(usb_path):
    cam = some_v4l_lib.Camera(usb_path)
    return cam


def save_video(camera, filename, codec):
    """Record video from camera to a file."""
    with open(filename, 'w') as out_file:
        video = VideoWriter(out_file, codec)
        for _ in range(1000):
            video.append(camera.get_frame())


main = cfgc.Config(save_video, camera=basic_camera, codec='libx264')

if __name__ == "__main__":
    cfgc.cli(main)
