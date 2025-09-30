from collections.abc import Iterator

import dearpygui.dearpygui as dpg
import numpy as np

import pimm

_DEFAULT_IMAGE_SHAPE = (240, 320)


def _get_down_keys() -> list[int]:
    all_keys = [getattr(dpg, key) for key in dir(dpg) if key.startswith('mvKey_')]
    return [key for key in all_keys if dpg.is_key_down(key)]


class DearpyguiUi(pimm.ControlSystem):
    def __init__(self):
        self.cameras = pimm.ReceiverDict(self)
        self.im_sizes = {}
        self.info = pimm.ControlSystemReceiver(self)
        self.buttons = pimm.ControlSystemEmitter(self)

    def init(self, im_sizes: dict[str, tuple[int, int]]):
        self.im_sizes = im_sizes
        self.n_rows = max(1, int(np.ceil(np.sqrt(len(self.cameras)))))
        self.n_cols = max(1, int(np.ceil(len(self.cameras) / self.n_rows)))

        if self.im_sizes:
            first_height, first_width = next(iter(self.im_sizes.values()))
        else:
            first_height, first_width = _DEFAULT_IMAGE_SHAPE
        self.im_height = first_height
        self.im_width = first_width

        self.raw_textures = {
            cam_name: np.ones((height, width, 4), dtype=np.float32)
            for cam_name, (height, width) in self.im_sizes.items()
        }

        dpg.create_context()
        with dpg.texture_registry():
            for cam_name, (height, width) in self.im_sizes.items():
                dpg.add_raw_texture(
                    width=width,
                    height=height,
                    tag=cam_name,
                    format=dpg.mvFormat_Float_rgba,
                    default_value=self.raw_textures[cam_name],
                )

        with dpg.window(tag='image_grid', label='Cameras', no_scrollbar=True, no_scroll_with_mouse=True):
            self._configure_image_grid(len(self.cameras))

        with dpg.window(label='Info'):
            dpg.add_text('', tag='info')

        with dpg.item_handler_registry(tag='adjust_images'):
            dpg.add_item_resize_handler(callback=self._configure_image_sizes)

        dpg.bind_item_handler_registry('image_grid', 'adjust_images')

        dpg.create_viewport(title='Positronic Viewer', width=800, height=600)
        dpg.set_viewport_resize_callback(callback=self._viewport_resize)
        dpg.setup_dearpygui()
        dpg.show_viewport(maximized=True)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        cameras = {
            cam_name: pimm.DefaultReceiver(pimm.ValueUpdated(reader), (None, False))
            for cam_name, reader in self.cameras.items()
        }

        fps_counter = pimm.utils.RateCounter('UI')
        frame_fps_counter = pimm.utils.RateCounter('Frame')

        info_receiver = pimm.DefaultReceiver(self.info, '')

        im_sizes = {}
        init_done = False

        if not init_done and len(cameras) == 0:
            self.init({})
            init_done = True

        while not should_stop.value and (not init_done or dpg.is_dearpygui_running()):
            if init_done:
                fps_counter.tick()
                pressed_keys = _get_down_keys()
                self.buttons.emit(pimm.Message(pressed_keys))

            for cam_name, camera in cameras.items():
                frame, is_new = camera.value

                if frame is not None and is_new:
                    image = frame['image']
                    if cam_name not in im_sizes:
                        im_sizes[cam_name] = image.shape[:2]
                        print(f'Have {len(im_sizes)}/{len(cameras)} images')
                        if not init_done and len(im_sizes) == len(cameras):
                            self.init(im_sizes)
                            init_done = True

                    if init_done:
                        texture = self.raw_textures[cam_name]
                        texture[:, :, :3] = image
                        texture[:, :, :3] /= 255
                    frame_fps_counter.tick()

            if init_done:
                info_text = info_receiver.value
                dpg.set_value('info', info_text)
                dpg.render_dearpygui_frame()

            yield pimm.Pass()

        print('GUI stopped')

    def _configure_image_grid(self, n_images: int):
        if n_images == 0:
            return

        with dpg.table(header_row=False):
            for _ in range(self.n_cols):
                dpg.add_table_column()

            for i in range(self.n_rows):
                with dpg.table_row():
                    for j in range(self.n_cols):
                        idx = i * self.n_cols + j
                        if idx < n_images:
                            cam_name = list(self.cameras.keys())[idx]
                            dpg.add_image(texture_tag=cam_name, tag=f'image_{cam_name}')

    def _viewport_resize(self):
        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()

        dpg.set_item_width('image_grid', width)
        dpg.set_item_height('image_grid', height)

    def _configure_image_sizes(self):
        n_images = len(self.cameras.keys())
        n_cols = max(1, int(np.ceil(np.sqrt(n_images))))
        n_rows = max(1, int(np.ceil(n_images / n_cols)))

        width = dpg.get_item_width('image_grid')
        height = dpg.get_item_height('image_grid')

        for key in self.cameras.keys():
            dpg.set_item_width(f'image_{key}', int(width / n_cols))
            dpg.set_item_height(f'image_{key}', int(height / n_rows))
