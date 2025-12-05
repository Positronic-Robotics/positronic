import json
import time
from collections.abc import Iterator
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


TASKS = [
    'Pick all the towels one by one from transparent tote and place them into the large grey tote.',
    'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.',
    'Pick all the scissors one by one from transparent tote and place them into the large grey tote.',
    'Pick up the green cube and put in on top of the red cube.',
    'Pick up objects from the red tote and place them in the green tote.',
]


class EvalUI(pimm.ControlSystem):
    def __init__(self, max_im_size: tuple[int, int] = (320, 240), ui_scale: float = 1.0):
        self.state = State.WAITING
        self.ui_scale = ui_scale
        self.max_im_size = (self.size(max_im_size[0]), self.size(max_im_size[1]))

        # --- Inputs/Outputs ---
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.inference_command = pimm.ControlSystemEmitter(self)
        self.ds_writer_command = pimm.ControlSystemEmitter(self)

        # UI State
        self.element_states: dict[str | int, list[State]] = {}

        # Internal state for camera rendering
        self.im_sizes = {}
        self.raw_textures = {}

        # New state
        self.cap_per_item = 30
        self.run_start_time = None
        self.run_duration = None

    def size(self, v: int) -> int:
        """Scale a value by ui_scale."""
        return int(v * self.ui_scale)

    def _register(self, tag: str | int, enabled_states: list[State]) -> str | int:
        """Registers a UI element's enabled states."""
        self.element_states[tag] = enabled_states
        return tag

    def _create_theme(self):
        with dpg.theme(tag='disabled_theme'):
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

        with dpg.theme(tag='finished_theme'):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 100, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 120, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 80, 0))

        with dpg.theme(tag='stall_theme'):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 150, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170, 170, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (130, 130, 0))

        with dpg.theme(tag='safety_theme'):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (130, 0, 0))

        with dpg.theme(tag='system_theme'):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (80, 80, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (100, 100, 100))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (60, 60, 60))

    def _build_controls(self):
        with dpg.group(horizontal=True):
            self._register(
                dpg.add_button(label='Start', callback=self.start, width=self.size(80), height=self.size(32)),
                [State.WAITING],
            )
            dpg.add_spacer(width=self.size(10))
            dpg.add_spacer(width=self.size(10))

            # Stop buttons
            btn_finished = dpg.add_button(
                label='Finished', callback=lambda: self.stop_run('Success'), width=self.size(80), height=self.size(32)
            )
            dpg.bind_item_theme(btn_finished, 'finished_theme')
            self._register(btn_finished, [State.RUNNING])

            dpg.add_spacer(width=self.size(5))
            btn_stall = dpg.add_button(
                label='Stall', callback=lambda: self.stop_run('Stalled'), width=self.size(80), height=self.size(32)
            )
            dpg.bind_item_theme(btn_stall, 'stall_theme')
            self._register(btn_stall, [State.RUNNING])

            dpg.add_spacer(width=self.size(5))
            btn_safety = dpg.add_button(
                label='Safety', callback=lambda: self.stop_run('Safety'), width=self.size(80), height=self.size(32)
            )
            dpg.bind_item_theme(btn_safety, 'safety_theme')
            self._register(btn_safety, [State.RUNNING])

            dpg.add_spacer(width=self.size(5))
            btn_system = dpg.add_button(
                label='System', callback=lambda: self.stop_run('System'), width=self.size(80), height=self.size(32)
            )
            dpg.bind_item_theme(btn_system, 'system_theme')
            self._register(btn_system, [State.RUNNING])
            dpg.add_spacer(width=self.size(10))
            self._register(
                dpg.add_button(label='Reset', callback=self.reset, width=self.size(80), height=self.size(32)),
                [State.WAITING, State.RUNNING],
            )

    def _build_configuration(self):
        dpg.add_text('Configuration')
        with dpg.group(horizontal=True):
            with dpg.child_window(height=self.size(180), width=self.size(480), border=True):
                dpg.add_text('Task')
                self._register(
                    dpg.add_radio_button(
                        items=[*TASKS, 'Other'], default_value=TASKS[0], callback=self.radio_callback, tag='task_radio'
                    ),
                    [State.WAITING],
                )
                dpg.add_spacer(height=self.size(5))
                self._register(
                    dpg.add_input_text(show=False, width=self.size(350), tag='custom_input'), [State.WAITING]
                )

            with dpg.group():
                self._register(
                    dpg.add_input_int(
                        label='Total items',
                        step=1,
                        width=self.size(100),
                        tag='total_items_input',
                        callback=self.validate_items_callback,
                    ),
                    [State.WAITING],
                )
                dpg.add_spacer(height=self.size(5))

                self._register(
                    dpg.add_input_int(
                        label='Successful items',
                        step=1,
                        width=self.size(100),
                        tag='successful_items_input',
                        callback=self.validate_items_callback,
                    ),
                    [State.RUNNING, State.REVIEWING],
                )

                dpg.add_spacer(height=self.size(5))
                self._register(
                    dpg.add_input_int(
                        label='Cap/item (s)',
                        default_value=self.cap_per_item,
                        width=self.size(100),
                        tag='cap_per_item_input',
                        callback=self.cap_callback,
                        step=1,
                    ),
                    [State.WAITING],
                )

                dpg.add_spacer(height=self.size(5))
                dpg.add_text('Total run cap: 0 sec', tag='total_run_cap_text')

                dpg.add_spacer(height=self.size(10))
                dpg.add_text('Tote Placement')
                self._register(
                    dpg.add_radio_button(
                        items=['left', 'right', 'NA'], default_value='NA', horizontal=True, tag='tote_radio'
                    ),
                    [State.WAITING, State.RUNNING, State.REVIEWING],
                )

                dpg.add_text('External Camera')
                self._register(
                    dpg.add_radio_button(
                        items=['left', 'right', 'NA'], default_value='NA', horizontal=True, tag='camera_radio'
                    ),
                    [State.WAITING, State.RUNNING, State.REVIEWING],
                )

        dpg.add_spacer(height=self.size(10))
        dpg.add_text('Outcome')
        self._register(
            dpg.add_radio_button(
                items=['Success', 'Stalled', 'Ran out of time', 'Safety', 'System'],
                default_value='Success',
                tag='outcome_radio',
            ),
            [State.REVIEWING],
        )

    def _build_notes(self):
        dpg.add_text('Notes')
        dpg.add_spacer(height=self.size(5))
        with dpg.group(horizontal=True):
            self._register(
                dpg.add_input_text(multiline=True, height=self.size(60), width=self.size(380), tag='notes_input'),
                [State.REVIEWING],
            )
            dpg.add_spacer(width=self.size(10))
            with dpg.group(horizontal=False):
                self._register(
                    dpg.add_button(label='Submit', width=self.size(100), height=self.size(28), callback=self.submit),
                    [State.REVIEWING],
                )
                dpg.add_spacer(height=self.size(5))
                self._register(
                    dpg.add_button(label='Cancel', width=self.size(100), height=self.size(28), callback=self.cancel),
                    [State.REVIEWING],
                )

    def _setup_key_handlers(self):
        with dpg.handler_registry():

            def safe_trigger(callback):
                text_inputs = ['notes_input', 'custom_input', 'total_items_input', 'successful_items_input']
                for tag in text_inputs:
                    if dpg.is_item_focused(tag):
                        return
                callback()

            dpg.add_key_press_handler(dpg.mvKey_S, callback=lambda s, a: safe_trigger(self.start))
            dpg.add_key_press_handler(dpg.mvKey_P, callback=lambda s, a: safe_trigger(lambda: self.stop_run('System')))
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

            def submit_on_enter():
                if self.state == State.REVIEWING:
                    self.submit()

            def cancel_on_escape():
                if self.state == State.REVIEWING:
                    self.cancel()

            dpg.add_key_press_handler(dpg.mvKey_Return, callback=lambda s, a: safe_trigger(submit_on_enter))
            dpg.add_key_press_handler(dpg.mvKey_Escape, callback=lambda s, a: safe_trigger(cancel_on_escape))

    # --- State Transitions ---

    def start(self, sender=None, app_data=None):
        if self.state != State.WAITING:
            return
        print('State: RUNNING')
        self.state = State.RUNNING
        self.update_ui()

        # Emit commands
        task_name = (
            dpg.get_value('custom_input') if dpg.get_value('task_radio') == 'Other' else dpg.get_value('task_radio')
        )
        self.run_start_time = self.clock.now()
        self.inference_command.emit(InferenceCommand.START(task=task_name))
        self.ds_writer_command.emit(DsWriterCommand.START(static_data={'task': task_name}))

    def stop_run(self, reason):
        if self.state != State.RUNNING:
            return
        print(f'State: REVIEWING ({reason})')
        self.state = State.REVIEWING
        self.run_duration = self.clock.now() - self.run_start_time

        dpg.set_value('outcome_radio', reason)
        if reason == 'Success':
            total = dpg.get_value('total_items_input')
            dpg.set_value('successful_items_input', total)

        self.update_ui()

        # Emit commands
        self.inference_command.emit(InferenceCommand.STOP())
        self.ds_writer_command.emit(DsWriterCommand.SUSPEND())

    def stop(self, sender=None, app_data=None):
        self.stop_run('System')

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
            'eval.outcome': dpg.get_value('outcome_radio'),
            'eval.notes': dpg.get_value('notes_input'),
            'eval.duration': self.run_duration,
            'eval.cap_per_item': self.cap_per_item,
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

    def cap_callback(self, sender, app_data):
        self.cap_per_item = app_data
        self.update_ui()

    def aborted_callback(self, sender, app_data):
        # Deprecated but kept if needed for other logic, though unused now
        pass

    def validate_items_callback(self, sender, app_data):
        total = dpg.get_value('total_items_input')
        successful = dpg.get_value('successful_items_input')

        if successful > total:
            dpg.set_value('successful_items_input', total)

        self.update_ui()

    # --- UI Update ---

    def update_ui(self, task_value=None):
        for tag, enabled_states in self.element_states.items():
            should_be_enabled = self.state in enabled_states
            if not should_be_enabled:
                dpg.bind_item_theme(tag, 'disabled_theme')
                dpg.configure_item(tag, enabled=False)
            else:
                dpg.bind_item_theme(tag, 0)
                dpg.configure_item(tag, enabled=True)

        # Special logic for dependent widgets
        if task_value is None:
            task_value = dpg.get_value('task_radio')

        is_other = task_value == 'Other'
        dpg.configure_item('custom_input', show=is_other)

        if self.state == State.WAITING:
            dpg.set_value('successful_items_input', 0)
            dpg.set_value('notes_input', '')

        # Update total run cap text
        total_items = dpg.get_value('total_items_input')
        total_seconds = total_items * self.cap_per_item
        mins = total_seconds // 60
        secs = total_seconds % 60
        if mins > 0:
            text = f'Total run cap: {mins} min {secs} sec'
        else:
            text = f'Total run cap: {secs} sec'
        dpg.set_value('total_run_cap_text', text)

    # --- Control System Run Loop ---

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self.clock = clock
        # Initialize DPG Context
        dpg.create_context()
        self._create_theme()

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
                    self._build_controls()

                    dpg.add_spacer(height=self.size(15))
                    dpg.add_separator()
                    dpg.add_spacer(height=self.size(10))

                    self._build_configuration()

                    dpg.add_spacer(height=self.size(15))
                    dpg.add_separator()
                    dpg.add_spacer(height=self.size(10))

                    self._build_notes()

        self._setup_key_handlers()

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
            # Check for time limit
            if self.state == State.RUNNING and self.run_start_time:
                elapsed = self.clock.now() - self.run_start_time
                total_cap = dpg.get_value('total_items_input') * self.cap_per_item
                if elapsed > total_cap:
                    self.stop_run('Ran out of time')

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
