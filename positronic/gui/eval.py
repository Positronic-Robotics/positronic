import json
import time
from collections.abc import Callable, Iterator
from enum import Enum, auto

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import pimm
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.policy.inference import InferenceCommand


class State(Enum):
    WAITING = auto()
    RUNNING = auto()
    REVIEWING = auto()


class UIElement:
    def __init__(self, tag: str, enabled_states: list[State], render_func: Callable, **kwargs):
        self.tag = tag
        self.enabled_states = enabled_states
        self.render_func = render_func
        self.kwargs = kwargs

    def render(self, **overrides):
        """Renders the DPG element."""
        kwargs = {**self.kwargs, **overrides}
        self.render_func(tag=self.tag, **kwargs)

    def update(self, current_state: State):
        """Updates the enabled/disabled state based on current_state."""
        should_be_enabled = current_state in self.enabled_states

        if not should_be_enabled:
            dpg.bind_item_theme(self.tag, 'disabled_theme')
            dpg.configure_item(self.tag, enabled=False)
        else:
            dpg.bind_item_theme(self.tag, 0)
            dpg.configure_item(self.tag, enabled=True)


class EvalUI(pimm.ControlSystem):
    def __init__(self, max_im_size: tuple[int, int] = (320, 240), ui_scale: float = 1.0):
        self.state = State.WAITING
        self.elements: list[UIElement] = []
        # Scale max_im_size by ui_scale
        self.ui_scale = ui_scale
        self.max_im_size = (self.size(max_im_size[0]), self.size(max_im_size[1]))

        # --- Inputs/Outputs ---
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.inference_command = pimm.ControlSystemEmitter(self)
        self.ds_writer_command = pimm.ControlSystemEmitter(self)

        # --- Actions ---
        self.start_btn = self._add(
            'start_btn',
            [State.WAITING],
            dpg.add_button,
            label='Start',
            callback=self.start,
            width=self.size(80),
            height=self.size(32),
        )
        self.stop_btn = self._add(
            'stop_btn',
            [State.RUNNING],
            dpg.add_button,
            label='Stop',
            callback=self.stop,
            width=self.size(80),
            height=self.size(32),
        )
        self.reset_btn = self._add(
            'reset_btn',
            [State.WAITING, State.RUNNING],
            dpg.add_button,
            label='Reset',
            callback=self.reset,
            width=self.size(80),
            height=self.size(32),
        )

        # --- Configuration ---
        tasks = [
            'Pick all the towels one by one from transparent tote and place them into the large grey tote.',
            'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.',
            'Pick all the scissors one by one from transparent tote and place them into the large grey tote.',
            'Pick up the green cube and put in on top of the red cube.',
            'Pick up objects from the red tote and place them in the green tote.',
        ]
        self.task_radio = self._add(
            'task_radio',
            [State.WAITING],
            dpg.add_radio_button,
            items=[*tasks, 'Other'],
            default_value=tasks[0],
            callback=self.radio_callback,
        )
        self.custom_input = self._add(
            'custom_input', [State.WAITING], dpg.add_input_text, show=False, width=self.size(350)
        )

        self.tote_radio = self._add(
            'tote_radio',
            [State.WAITING, State.RUNNING, State.REVIEWING],
            dpg.add_radio_button,
            items=['left', 'right', 'NA'],
            default_value='NA',
            horizontal=True,
        )
        self.camera_radio = self._add(
            'camera_radio',
            [State.WAITING, State.RUNNING, State.REVIEWING],
            dpg.add_radio_button,
            items=['left', 'right', 'NA'],
            default_value='NA',
            horizontal=True,
        )

        self.total_items = self._add(
            'total_items_input',
            [State.WAITING, State.REVIEWING],
            dpg.add_input_int,
            label='Total items',
            step=1,
            width=self.size(100),
        )
        self.successful_items = self._add(
            'successful_items_input',
            [State.RUNNING, State.REVIEWING],
            dpg.add_input_int,
            label='Successful items',
            step=1,
            width=self.size(100),
        )

        self.aborted_checkbox = self._add(
            'aborted_checkbox', [State.REVIEWING], dpg.add_checkbox, label='Aborted', callback=self.aborted_callback
        )
        self.model_failure_checkbox = self._add(
            'model_failure_checkbox',
            [State.REVIEWING],
            dpg.add_checkbox,
            label='Model failure',
            callback=self.model_failure_callback,
        )

        # --- Notes & Submission ---
        self.notes_input = self._add(
            'notes_input',
            [State.REVIEWING],
            dpg.add_input_text,
            multiline=True,
            height=self.size(60),
            width=self.size(380),
        )
        self.submit_btn = self._add(
            'submit_btn',
            [State.REVIEWING],
            dpg.add_button,
            label='Submit',
            width=self.size(100),
            height=self.size(28),
            callback=self.submit,
        )
        self.cancel_btn = self._add(
            'cancel_btn',
            [State.REVIEWING],
            dpg.add_button,
            label='Cancel',
            width=self.size(100),
            height=self.size(28),
            callback=self.cancel,
        )

        # Internal state for camera rendering
        self.im_sizes = {}
        self.raw_textures = {}
        self.n_rows = 1
        self.n_cols = 1

    def _add(self, tag, enabled_states, render_func, **kwargs):
        element = UIElement(tag, enabled_states, render_func, **kwargs)
        self.elements.append(element)
        return element

    def size(self, v: int) -> int:
        """Scale a value by ui_scale."""
        return int(v * self.ui_scale)

    # --- State Transitions ---

    def start(self, sender=None, app_data=None):
        if self.state != State.WAITING:
            return
        print('State: RUNNING')
        self.state = State.RUNNING
        self.update_ui()

        # Emit commands
        task_name = (
            dpg.get_value(self.custom_input.tag)
            if dpg.get_value(self.task_radio.tag) == 'Other'
            else dpg.get_value(self.task_radio.tag)
        )
        self.inference_command.emit(InferenceCommand.START(task=task_name))
        self.ds_writer_command.emit(DsWriterCommand.START(static_data={'task': task_name}))

    def stop(self, sender=None, app_data=None):
        if self.state != State.RUNNING:
            return
        print('State: REVIEWING')
        self.state = State.REVIEWING
        self.update_ui()

        # Emit commands
        self.inference_command.emit(InferenceCommand.STOP())
        self.ds_writer_command.emit(DsWriterCommand.SUSPEND())

    def reset(self, sender=None, app_data=None):
        if self.state == State.REVIEWING:
            return  # Reset disabled in REVIEWING

        print('State: WAITING (Reset)')
        # Reset data
        dpg.set_value('successful_items_input', 0)
        # Reset to WAITING
        self.state = State.WAITING
        self.update_ui()

        # Emit commands
        self.inference_command.emit(InferenceCommand.RESET())

    def submit(self, sender=None, app_data=None):
        if self.state != State.REVIEWING:
            return

        data = {
            'eval.total_items': dpg.get_value('total_items_input'),
            'eval.successful_items': dpg.get_value('successful_items_input'),
            'eval.aborted': dpg.get_value('aborted_checkbox'),
            'eval.model_failure': dpg.get_value('model_failure_checkbox'),
            'eval.notes': dpg.get_value('notes_input'),
        }

        # Add conditional fields
        tote_val = dpg.get_value('tote_radio')
        if tote_val != 'NA':
            data['eval.tote_placement'] = tote_val

        camera_val = dpg.get_value('camera_radio')
        if camera_val != 'NA':
            data['eval.external_camera'] = camera_val

        print(json.dumps(data, indent=2))

        dpg.set_value('notes_input', '')
        dpg.set_value('successful_items_input', 0)
        self.state = State.WAITING
        self.update_ui()

        # Emit commands
        self.ds_writer_command.emit(DsWriterCommand.STOP(static_data=data))

    def cancel(self, sender=None, app_data=None):
        if self.state != State.REVIEWING:
            return
        print('State: WAITING (Cancelled)')
        dpg.set_value('notes_input', '')
        dpg.set_value('successful_items_input', 0)
        self.state = State.WAITING
        self.update_ui()

        # Emit commands
        self.ds_writer_command.emit(DsWriterCommand.ABORT())

    # --- Callbacks ---

    def radio_callback(self, sender, app_data):
        # Guard: Only allow change in WAITING
        if self.state != State.WAITING:
            return

        # We need to update UI to ensure the new custom input gets correct state
        self.update_ui(task_value=app_data)

    def aborted_callback(self, sender, app_data):
        if self.state != State.REVIEWING:
            dpg.set_value('aborted_checkbox', False)
        self.update_ui()

    def model_failure_callback(self, sender, app_data):
        pass

    # --- UI Update ---

    def update_ui(self, task_value=None):
        for element in self.elements:
            element.update(self.state)

        # Special logic for dependent widgets
        if task_value is None:
            task_value = dpg.get_value('task_radio')

        is_other = task_value == 'Other'
        dpg.configure_item(self.custom_input.tag, show=is_other)

        if self.state == State.REVIEWING:
            if not dpg.get_value('aborted_checkbox'):
                # We can't use _set_fake_disabled anymore, use direct configure
                dpg.configure_item(self.model_failure_checkbox.tag, enabled=False)
                dpg.set_value(self.model_failure_checkbox.tag, False)
            else:
                dpg.configure_item(self.model_failure_checkbox.tag, enabled=True)
        else:
            dpg.configure_item(self.model_failure_checkbox.tag, enabled=False)
            dpg.set_value(self.model_failure_checkbox.tag, False)

        if self.state == State.WAITING:
            dpg.set_value(self.successful_items.tag, 0)
            dpg.set_value(self.aborted_checkbox.tag, False)
            dpg.set_value(self.model_failure_checkbox.tag, False)
            dpg.set_value(self.notes_input.tag, '')

    # --- Control System Run Loop ---

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        # Initialize DPG Context
        dpg.create_context()

        # Initialize textures based on available cameras (wait for first frame or default)
        # For now, we'll initialize dynamically in the loop or just setup a placeholder if needed.
        # But DPG needs textures created before adding images usually, or added dynamically.
        # Let's use a dynamic approach similar to dpg.py but adapted.

        # General Theme for "Fake Disabled" Elements
        with dpg.theme(tag='disabled_theme'):
            # Default state (fallback)
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (20, 20, 20))

            # Explicit disabled state
            with dpg.theme_component(dpg.mvAll, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (20, 20, 20))

            # Specific components (Default)
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (100, 100, 100))

            with dpg.theme_component(dpg.mvRadioButton):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 100, 100))

            # Specific components (Disabled)
            with dpg.theme_component(dpg.mvButton, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (100, 100, 100))

            with dpg.theme_component(dpg.mvRadioButton, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 100, 100))

        # Window
        with dpg.window(label='Evaluation Control', width=self.size(1200), height=self.size(800), tag='main_window'):
            with dpg.group(horizontal=True):
                # Left side: Camera feeds in vertical column
                with dpg.group(horizontal=False, tag='image_grid_group'):
                    dpg.add_text('Camera Feed')

                # Spacer between images and controls
                dpg.add_spacer(width=self.size(20))

                # Right side: Controls
                with dpg.group(horizontal=False):
                    dpg.add_text('Controls')
                    dpg.add_spacer(height=self.size(5))
                    with dpg.group(horizontal=True):
                        self.start_btn.render()
                        dpg.add_spacer(width=self.size(10))
                        self.stop_btn.render()
                        dpg.add_spacer(width=self.size(10))
                        self.reset_btn.render()

                    dpg.add_spacer(height=self.size(15))
                    dpg.add_separator()
                    dpg.add_spacer(height=self.size(10))
                    dpg.add_text('Configuration')

                    with dpg.group(horizontal=True):
                        # Increased height to fit vertical stack of radio buttons + input
                        with dpg.child_window(height=self.size(180), width=self.size(480), border=True):
                            dpg.add_text('Task')
                            # Vertical layout: Radio buttons first, then input below
                            self.task_radio.render()
                            dpg.add_spacer(height=self.size(5))
                            self.custom_input.render()

                        with dpg.group():
                            self.total_items.render()
                            dpg.add_spacer(height=self.size(5))
                            self.successful_items.render()

                            dpg.add_spacer(height=self.size(10))
                            dpg.add_text('Tote Placement')
                            self.tote_radio.render()

                            dpg.add_text('External Camera')
                            self.camera_radio.render()

                    dpg.add_spacer(height=self.size(10))
                    self.aborted_checkbox.render()
                    dpg.add_spacer(height=self.size(5))
                    self.model_failure_checkbox.render()

                    dpg.add_spacer(height=self.size(15))
                    dpg.add_separator()
                    dpg.add_spacer(height=self.size(10))
                    dpg.add_text('Notes')
                    dpg.add_spacer(height=self.size(5))
                    with dpg.group(horizontal=True):
                        self.notes_input.render()
                        dpg.add_spacer(width=self.size(10))
                        with dpg.group(horizontal=False):
                            self.submit_btn.render()
                            dpg.add_spacer(height=self.size(5))
                            self.cancel_btn.render()

        # Key Handler
        with dpg.handler_registry():

            def safe_trigger(callback):
                text_inputs = ['notes_input', 'custom_input', 'total_items_input', 'successful_items_input']
                for tag in text_inputs:
                    if dpg.is_item_focused(tag):
                        return
                callback()

            dpg.add_key_press_handler(dpg.mvKey_S, callback=lambda s, a: safe_trigger(self.start))
            dpg.add_key_press_handler(dpg.mvKey_P, callback=lambda s, a: safe_trigger(self.stop))
            dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda s, a: safe_trigger(self.reset))

            def change_radio_selection(sender, app_data):
                if self.state != State.WAITING:
                    return
                if dpg.is_item_focused('task_radio'):
                    items = dpg.get_item_configuration('task_radio')['items']
                    current = dpg.get_value('task_radio')
                    idx = items.index(current)
                    if app_data == dpg.mvKey_Up:
                        new_idx = max(0, idx - 1)
                    elif app_data == dpg.mvKey_Down:
                        new_idx = min(len(items) - 1, idx + 1)
                    else:
                        return
                    new_value = items[new_idx]
                    dpg.set_value('task_radio', new_value)
                    self.radio_callback('task_radio', new_value)

            dpg.add_key_press_handler(dpg.mvKey_Up, callback=change_radio_selection)
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=change_radio_selection)

            # Add key handlers for submit/cancel when in REVIEWING state
            def submit_on_enter():
                if self.state == State.REVIEWING:
                    self.submit()

            def cancel_on_escape():
                if self.state == State.REVIEWING:
                    self.cancel()

            dpg.add_key_press_handler(dpg.mvKey_Return, callback=lambda s, a: safe_trigger(submit_on_enter))
            dpg.add_key_press_handler(dpg.mvKey_Escape, callback=lambda s, a: safe_trigger(cancel_on_escape))

        dpg.create_viewport(title='Eval UI', width=self.size(520), height=self.size(850))
        dpg.set_viewport_vsync(True)
        dpg.configure_app(keyboard_navigation=True)
        if self.ui_scale != 1.0:
            dpg.set_global_font_scale(self.ui_scale)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()

        # Initialize UI state
        self.update_ui()

        while not should_stop.value and dpg.is_dearpygui_running():
            # Handle Cameras
            for cam_name, camera in self.cameras.items():
                cam_msg = camera.read()
                if cam_msg.data is not None and cam_msg.updated:
                    image = cam_msg.data.array

                    if cam_name not in self.im_sizes:
                        orig_height, orig_width = image.shape[:2]

                        # Calculate display size (downsample if needed)
                        max_width, max_height = self.max_im_size
                        scale = min(max_width / orig_width, max_height / orig_height, 1.0)
                        display_width = int(orig_width * scale)
                        display_height = int(orig_height * scale)

                        self.im_sizes[cam_name] = (display_height, display_width)

                        with dpg.texture_registry(show=False):
                            data = np.zeros((display_height, display_width, 4), dtype=np.float32)
                            dpg.add_raw_texture(
                                display_width,
                                display_height,
                                default_value=data,
                                format=dpg.mvFormat_Float_rgba,
                                tag=f'tex_{cam_name}',
                            )
                            self.raw_textures[cam_name] = data

                        dpg.add_image(
                            f'tex_{cam_name}', parent='image_grid_group', width=display_width, height=display_height
                        )

                    # Downsample image if needed to match display size
                    display_height, display_width = self.im_sizes[cam_name]
                    if image.shape[0] != display_height or image.shape[1] != display_width:
                        image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

                    texture = self.raw_textures[cam_name]
                    texture[:, :, :3] = image / 255.0
                    texture[:, :, 3] = 1.0

            dpg.render_dearpygui_frame()
            yield pimm.Pass()

        dpg.destroy_context()


def main():
    import numpy as np

    class FakeCamera(pimm.ControlSystem):
        def __init__(self):
            super().__init__()
            self.frame = pimm.ControlSystemEmitter(self)

        def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
            adapter = None
            while not should_stop.value:
                img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                adapter = pimm.shared_memory.NumpySMAdapter.lazy_init(img, adapter)
                self.frame.emit(adapter)
                yield pimm.Sleep(1 / 15)

    with pimm.World() as world:
        ui = EvalUI()
        fake_camera = FakeCamera()
        world.connect(fake_camera.frame, ui.cameras['main'])

        for cmd in world.start([ui, fake_camera]):
            time.sleep(cmd.seconds)


if __name__ == '__main__':
    main()
