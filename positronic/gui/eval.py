import json
from collections.abc import Callable
from enum import Enum, auto

import dearpygui.dearpygui as dpg
import numpy as np


def create_mock_image(width, height):
    # Generate random RGBA noise (0.0 to 1.0)
    data = np.random.random((height, width, 4))
    return data.flatten()


def on_start():
    print('Start triggered')


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


class EvalController:
    def __init__(self):
        self.state = State.WAITING
        self.elements: list[UIElement] = []

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

    def stop(self):
        if self.state != State.RUNNING:
            return
        print('State: REVIEWING')
        self.state = State.REVIEWING
        self.update_ui()

    def reset(self):
        if self.state == State.REVIEWING:
            return  # Reset disabled in REVIEWING

        print('State: WAITING (Reset)')
        # Reset data
        dpg.set_value('successful_items_input', 0)
        # Reset to WAITING
        self.state = State.WAITING
        self.update_ui()

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

    def cancel(self):
        if self.state != State.REVIEWING:
            return
        print('State: WAITING (Cancelled)')
        dpg.set_value('notes_input', '')
        dpg.set_value('successful_items_input', 0)
        self.state = State.WAITING
        self.update_ui()

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
        # This callback is only triggered if the checkbox is interacted with.
        # The actual enabled/disabled state is managed by update_ui.
        # If it's clicked when it shouldn't be, update_ui will correct it.
        pass

    # --- UI Update ---

    def update_ui(self):
        for element in self.elements:
            element.update(self.state)

        # Special logic for dependent widgets
        # Custom Input: Visibility depends on 'Other' being selected
        is_other = dpg.get_value('task_radio') == 'Other'
        dpg.configure_item(self.custom_input.tag, show=is_other)

        # Model Failure: Only enabled if Aborted is checked AND we are in REVIEWING
        if self.state == State.REVIEWING:
            if not dpg.get_value('aborted_checkbox'):
                self.model_failure_checkbox._set_fake_disabled(True)
                dpg.set_value(self.model_failure_checkbox.tag, False)  # Also reset value if disabled
            else:
                # If aborted is checked, it should follow its base enabled_states (which is REVIEWING)
                self.model_failure_checkbox._set_fake_disabled(False)
        else:
            # If not in REVIEWING, it should be disabled based on its base config
            self.model_failure_checkbox._set_fake_disabled(True)
            dpg.set_value(self.model_failure_checkbox.tag, False)  # Also reset value if disabled

        # Reset values for certain items when state changes to WAITING
        if self.state == State.WAITING:
            dpg.set_value(self.successful_items.tag, 0)
            dpg.set_value(self.aborted_checkbox.tag, False)
            dpg.set_value(self.model_failure_checkbox.tag, False)
            dpg.set_value(self.notes_input.tag, '')


controller = EvalController()


def main():
    dpg.create_context()

    # Mock Image Data
    width, height, channels = 256, 256, 3
    texture_data = np.random.random((width, height, channels)).flatten()

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width, height=height, default_value=texture_data, tag='mock_image')

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
    with dpg.window(label='Evaluation Control', width=500, height=800):
        dpg.add_text('Mock Image Feed')
        dpg.add_image('mock_image')

        dpg.add_separator()
        dpg.add_text('Controls')
        with dpg.group(horizontal=True):
            controller.start_btn.render()
            controller.stop_btn.render()
            controller.reset_btn.render()

        dpg.add_separator()
        dpg.add_text('Configuration')

        # Split into two columns: Radio Group (Left) and Numeric Inputs (Right)
        with dpg.group(horizontal=True):
            # Left Column: Radio buttons with custom input
            with dpg.child_window(height=110, width=240, border=True):
                dpg.add_text('Task')
                with dpg.group(horizontal=True):
                    controller.task_radio.render()

                    with dpg.group():
                        # Spacer to push the input down to the 3rd radio option
                        dpg.add_spacer(height=42)
                        controller.custom_input.render()

            # Right Column: Numeric Inputs
            with dpg.group():
                controller.total_items.render()
                controller.successful_items.render()

        controller.aborted_checkbox.render()
        controller.model_failure_checkbox.render()
        # Initial binding done by update_ui

        dpg.add_separator()
        dpg.add_text('Notes')
        with dpg.group(horizontal=True):
            controller.notes_input.render()
            with dpg.group(horizontal=False):
                controller.submit_btn.render()
                controller.cancel_btn.render()

    # Key Handler
    with dpg.handler_registry():

        def safe_trigger(callback):
            text_inputs = ['notes_input', 'custom_input', 'total_items_input', 'successful_items_input']
            for tag in text_inputs:
                if dpg.is_item_focused(tag):
                    return  # Ignore shortcut if typing
            callback()

        dpg.add_key_press_handler(dpg.mvKey_S, callback=lambda s, a: safe_trigger(controller.start))
        dpg.add_key_press_handler(dpg.mvKey_P, callback=lambda s, a: safe_trigger(controller.stop))
        dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda s, a: safe_trigger(controller.reset))

        # Custom handler for Radio Button navigation (Up/Down arrows)
        def change_radio_selection(sender, app_data):
            # Only allow if radio is not disabled (check state)
            if controller.state != State.WAITING:
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
                controller.radio_callback('task_radio', new_value)

        dpg.add_key_press_handler(dpg.mvKey_Up, callback=change_radio_selection)
        dpg.add_key_press_handler(dpg.mvKey_Down, callback=change_radio_selection)

    dpg.create_viewport(title='Eval UI', width=520, height=650)
    dpg.configure_app(keyboard_navigation=True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Initialize UI state
    controller.update_ui()

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
