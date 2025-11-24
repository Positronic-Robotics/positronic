import json
import time
from collections.abc import Callable, Iterator
from enum import Enum, auto

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
        self._set_fake_disabled(not should_be_enabled)

    def _set_fake_disabled(self, is_disabled: bool):
        # Always keep enabled=True to allow custom styling
        dpg.configure_item(self.tag, enabled=True)

        if is_disabled:
            dpg.bind_item_theme(self.tag, 'disabled_theme')
            # For inputs, make them readonly if possible
            if dpg.get_item_type(self.tag) in ['mvAppItemType::mvInputText', 'mvAppItemType::mvInputInt']:
                dpg.configure_item(self.tag, readonly=True)
        else:
            dpg.bind_item_theme(self.tag, 0)  # Reset to default
            if dpg.get_item_type(self.tag) in ['mvAppItemType::mvInputText', 'mvAppItemType::mvInputInt']:
                dpg.configure_item(self.tag, readonly=False)


class EvalUI(pimm.ControlSystem):
    def __init__(self):
        self.state = State.WAITING
        self.elements: list[UIElement] = []

        # --- Inputs/Outputs ---
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.inference_command = pimm.ControlSystemEmitter(self)
        self.ds_writer_command = pimm.ControlSystemEmitter(self)

        # --- Actions ---
        self.start_btn = self._add('start_btn', [State.WAITING], dpg.add_button, label='Start', callback=self.start)
        self.stop_btn = self._add('stop_btn', [State.RUNNING], dpg.add_button, label='Stop', callback=self.stop)
        self.reset_btn = self._add(
            'reset_btn', [State.WAITING, State.RUNNING], dpg.add_button, label='Reset', callback=self.reset
        )

        # --- Configuration ---
        self.task_radio = self._add(
            'task_radio',
            [State.WAITING],
            dpg.add_radio_button,
            items=['Towels', 'Spoons', 'Other'],
            default_value='Towels',
            callback=self.radio_callback,
        )
        self.custom_input = self._add('custom_input', [State.WAITING], dpg.add_input_text, show=False, width=130)

        self.total_items = self._add(
            'total_items_input',
            [State.WAITING, State.REVIEWING],
            dpg.add_input_int,
            label='Total items',
            step=1,
            width=100,
        )
        self.successful_items = self._add(
            'successful_items_input',
            [State.RUNNING, State.REVIEWING],
            dpg.add_input_int,
            label='Successful items',
            step=1,
            width=100,
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
            'notes_input', [State.REVIEWING], dpg.add_input_text, multiline=True, height=60, width=380
        )
        self.submit_btn = self._add(
            'submit_btn',
            [State.REVIEWING],
            dpg.add_button,
            label='Submit',
            width=100,
            height=28,
            callback=lambda: self.submit(),
        )
        self.cancel_btn = self._add(
            'cancel_btn',
            [State.REVIEWING],
            dpg.add_button,
            label='Cancel',
            width=100,
            height=28,
            callback=lambda: self.cancel(),
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

    # --- State Transitions ---

    def start(self):
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
        self.inference_command.emit(InferenceCommand.START())
        self.ds_writer_command.emit(DsWriterCommand.START(static_data={'task': task_name}))

    def stop(self):
        if self.state != State.RUNNING:
            return
        print('State: REVIEWING')
        self.state = State.REVIEWING
        self.update_ui()

        # Emit commands
        self.inference_command.emit(InferenceCommand.STOP())
        self.ds_writer_command.emit(DsWriterCommand.SUSPEND())

    def reset(self):
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

    def submit(self):
        if self.state != State.REVIEWING:
            return

        data = {
            'task': dpg.get_value('task_radio'),
            'custom_task': dpg.get_value('custom_input') if dpg.get_value('task_radio') == 'Other' else None,
            'total_items': dpg.get_value('total_items_input'),
            'successful_items': dpg.get_value('successful_items_input'),
            'aborted': dpg.get_value('aborted_checkbox'),
            'model_failure': dpg.get_value('model_failure_checkbox'),
            'notes': dpg.get_value('notes_input'),
        }
        print(json.dumps(data, indent=2))

        dpg.set_value('notes_input', '')
        dpg.set_value('successful_items_input', 0)
        self.state = State.WAITING
        self.update_ui()

        # Emit commands
        self.ds_writer_command.emit(DsWriterCommand.STOP(static_data=data))

    def cancel(self):
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
            pass

        show_custom = app_data == 'Other'
        dpg.configure_item('custom_input', show=show_custom)
        # We need to update UI to ensure the new custom input gets correct state
        self.update_ui()

    def aborted_callback(self, sender, app_data):
        if self.state != State.REVIEWING:
            dpg.set_value('aborted_checkbox', False)
        self.update_ui()

    def model_failure_callback(self, sender, app_data):
        pass

    # --- UI Update ---

    def update_ui(self):
        for element in self.elements:
            element.update(self.state)

        # Special logic for dependent widgets
        is_other = dpg.get_value('task_radio') == 'Other'
        dpg.configure_item(self.custom_input.tag, show=is_other)

        if self.state == State.REVIEWING:
            if not dpg.get_value('aborted_checkbox'):
                self.model_failure_checkbox._set_fake_disabled(True)
                dpg.set_value(self.model_failure_checkbox.tag, False)
            else:
                self.model_failure_checkbox._set_fake_disabled(False)
        else:
            self.model_failure_checkbox._set_fake_disabled(True)
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
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (80, 80, 80))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (80, 80, 80))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (30, 30, 30))

        # Window
        with dpg.window(label='Evaluation Control', width=500, height=800, tag='main_window'):
            dpg.add_text('Camera Feed')
            # Image grid container
            with dpg.group(tag='image_grid_group'):
                pass  # Images will be added here

            dpg.add_separator()
            dpg.add_text('Controls')
            with dpg.group(horizontal=True):
                self.start_btn.render()
                self.stop_btn.render()
                self.reset_btn.render()

            dpg.add_separator()
            dpg.add_text('Configuration')

            with dpg.group(horizontal=True):
                with dpg.child_window(height=110, width=240, border=True):
                    dpg.add_text('Task')
                    with dpg.group(horizontal=True):
                        self.task_radio.render()
                        with dpg.group():
                            dpg.add_spacer(height=42)
                            self.custom_input.render()

                with dpg.group():
                    self.total_items.render()
                    self.successful_items.render()

            self.aborted_checkbox.render()
            self.model_failure_checkbox.render()

            dpg.add_separator()
            dpg.add_text('Notes')
            with dpg.group(horizontal=True):
                self.notes_input.render()
                with dpg.group(horizontal=False):
                    self.submit_btn.render()
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

        dpg.create_viewport(title='Eval UI', width=520, height=850)
        dpg.configure_app(keyboard_navigation=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Initialize UI state
        self.update_ui()

        while not should_stop.value and dpg.is_dearpygui_running():
            # Handle Cameras
            for cam_name, camera in self.cameras.items():
                cam_msg = camera.read()
                if cam_msg.data is not None and cam_msg.updated:
                    image = cam_msg.data.array

                    if cam_name not in self.im_sizes:
                        height, width = image.shape[:2]
                        self.im_sizes[cam_name] = (height, width)

                        with dpg.texture_registry(show=False):
                            data = np.zeros((height, width, 4), dtype=np.float32)
                            dpg.add_raw_texture(
                                width, height, default_value=data, format=dpg.mvFormat_Float_rgba, tag=f'tex_{cam_name}'
                            )
                            self.raw_textures[cam_name] = data

                        dpg.add_image(f'tex_{cam_name}', parent='image_grid_group', width=width, height=height)

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
