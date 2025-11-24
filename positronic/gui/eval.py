import json
from enum import Enum, auto

import dearpygui.dearpygui as dpg
import numpy as np


def create_mock_image(width, height):
    # Generate random RGBA noise (0.0 to 1.0)
    data = np.random.random((height, width, 4))
    return data.flatten()


def on_start():
    print('Start triggered')


class EvalState(Enum):
    WAITING = auto()
    RUNNING = auto()
    REVIEWING = auto()


class EvalController:
    def __init__(self):
        self.state = EvalState.WAITING

    def start(self):
        if self.state != EvalState.WAITING:
            return
        print('State: RUNNING')
        self.state = EvalState.RUNNING
        self.update_ui()

    def stop(self):
        if self.state != EvalState.RUNNING:
            return
        print('State: REVIEWING')
        self.state = EvalState.REVIEWING
        self.update_ui()

    def reset(self):
        if self.state == EvalState.WAITING:
            # Allow reset in WAITING to just reset data? Or only from RUNNING/REVIEWING?
            # Requirement: "Pressing reset issues reset in the console." in WAITING.
            pass
        elif self.state == EvalState.RUNNING:
            pass
        elif self.state == EvalState.REVIEWING:
            return  # Reset disabled in REVIEWING

        print('State: WAITING (Reset)')
        # Reset data
        dpg.set_value('successful_items_input', 0)
        # Reset to WAITING
        self.state = EvalState.WAITING
        self.update_ui()

    def submit(self):
        if self.state != EvalState.REVIEWING:
            return

        # Gather data
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

        # Clear data
        dpg.set_value('notes_input', '')
        dpg.set_value('successful_items_input', 0)

        # Transition to WAITING
        self.state = EvalState.WAITING
        self.update_ui()

    def cancel(self):
        if self.state != EvalState.REVIEWING:
            return

        print('State: WAITING (Cancelled)')

        # Clear data
        dpg.set_value('notes_input', '')
        dpg.set_value('successful_items_input', 0)

        # Transition to WAITING
        self.state = EvalState.WAITING
        self.update_ui()

    def update_ui(self):
        state = self.state

        # Helper for "Fake Disabled" styling
        def set_disabled(tag, is_disabled):
            # Always keep enabled=True to allow custom styling
            dpg.configure_item(tag, enabled=True)

            if is_disabled:
                dpg.bind_item_theme(tag, 'disabled_theme')
                # For inputs, make them readonly if possible
                if dpg.get_item_type(tag) in ['mvAppItemType::mvInputText', 'mvAppItemType::mvInputInt']:
                    dpg.configure_item(tag, readonly=True)
            else:
                dpg.bind_item_theme(tag, 0)  # Reset to default
                if dpg.get_item_type(tag) in ['mvAppItemType::mvInputText', 'mvAppItemType::mvInputInt']:
                    dpg.configure_item(tag, readonly=False)

        # WAITING
        if state == EvalState.WAITING:
            set_disabled('start_btn', False)
            set_disabled('stop_btn', True)
            set_disabled('reset_btn', False)

            set_disabled('task_radio', False)
            # Custom input enabled only if Other selected
            set_disabled('custom_input', dpg.get_value('task_radio') != 'Other')

            set_disabled('total_items_input', False)
            dpg.set_value('successful_items_input', 0)
            set_disabled('successful_items_input', True)

            dpg.set_value('aborted_checkbox', False)
            set_disabled('aborted_checkbox', True)
            set_disabled('model_failure_checkbox', True)

            dpg.set_value('notes_input', '')
            set_disabled('notes_input', True)
            set_disabled('submit_btn', True)
            set_disabled('cancel_btn', True)

        # RUNNING
        elif state == EvalState.RUNNING:
            set_disabled('start_btn', True)
            set_disabled('stop_btn', False)
            set_disabled('reset_btn', False)

            set_disabled('task_radio', True)
            set_disabled('custom_input', True)

            set_disabled('total_items_input', True)
            set_disabled('successful_items_input', False)

            set_disabled('aborted_checkbox', True)
            set_disabled('model_failure_checkbox', True)

            set_disabled('notes_input', True)
            set_disabled('submit_btn', True)
            set_disabled('cancel_btn', True)

        # REVIEWING
        elif state == EvalState.REVIEWING:
            set_disabled('start_btn', True)
            set_disabled('stop_btn', True)
            set_disabled('reset_btn', True)

            set_disabled('task_radio', True)
            set_disabled('custom_input', True)

            set_disabled('total_items_input', False)
            set_disabled('successful_items_input', False)

            set_disabled('aborted_checkbox', False)

            # Model failure depends on Aborted
            if dpg.get_value('aborted_checkbox'):
                set_disabled('model_failure_checkbox', False)
            else:
                set_disabled('model_failure_checkbox', True)

            set_disabled('notes_input', False)
            set_disabled('submit_btn', False)
            set_disabled('cancel_btn', False)


controller = EvalController()


def radio_callback(sender, app_data):
    # Guard: Only allow change in WAITING
    if controller.state != EvalState.WAITING:
        # Revert change? Hard to know previous value without storing it.
        # But since we set it to "fake disabled", user click shouldn't visually do much if we handle it right?
        pass

    show_custom = app_data == 'Other'
    dpg.configure_item('custom_input', show=show_custom)
    controller.update_ui()


def aborted_callback(sender, app_data):
    if controller.state != EvalState.REVIEWING:
        dpg.set_value('aborted_checkbox', False)  # Force off if not in reviewing
    controller.update_ui()


def model_failure_callback(sender, app_data):
    # Only allowed in REVIEWING and if Aborted is checked
    allowed = (controller.state == EvalState.REVIEWING) and dpg.get_value('aborted_checkbox')
    if not allowed:
        dpg.set_value('model_failure_checkbox', False)


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
            dpg.add_button(label='[S]tart', tag='start_btn', callback=lambda: controller.start())
            dpg.add_button(label='Sto[p]', tag='stop_btn', callback=lambda: controller.stop())
            dpg.add_button(label='[R]eset', tag='reset_btn', callback=lambda: controller.reset())

        dpg.add_separator()
        dpg.add_text('Configuration')

        # Split into two columns: Radio Group (Left) and Numeric Inputs (Right)
        with dpg.group(horizontal=True):
            # Left Column: Radio buttons with custom input
            with dpg.child_window(height=110, width=240, border=True):
                dpg.add_text('Task')
                with dpg.group(horizontal=True):
                    dpg.add_radio_button(
                        items=['Towels', 'Spoons', 'Other'],
                        callback=radio_callback,
                        default_value='Towels',
                        tag='task_radio',
                    )

                    with dpg.group():
                        # Spacer to push the input down to the 3rd radio option
                        dpg.add_spacer(height=42)
                        dpg.add_input_text(tag='custom_input', show=False, width=130)

            # Right Column: Numeric Inputs
            with dpg.group():
                dpg.add_input_int(label='Total items', step=1, width=100, tag='total_items_input')
                dpg.add_input_int(label='Successful items', step=1, width=100, tag='successful_items_input')

        # Checkbox
        dpg.add_checkbox(label='Aborted', tag='aborted_checkbox', callback=aborted_callback)
        dpg.add_checkbox(label='Model failure', tag='model_failure_checkbox', callback=model_failure_callback)
        # Initial binding done by update_ui

        dpg.add_separator()
        dpg.add_text('Notes')
        with dpg.group(horizontal=True):
            dpg.add_input_text(multiline=True, height=60, width=380, tag='notes_input')
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label='Submit', width=100, height=28, tag='submit_btn', callback=lambda: controller.submit()
                )
                dpg.add_button(
                    label='Cancel', width=100, height=28, tag='cancel_btn', callback=lambda: controller.cancel()
                )

    # Key Handler
    with dpg.handler_registry():

        def safe_trigger(callback):
            # List of text input tags to check for focus
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
            if controller.state != EvalState.WAITING:
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
                radio_callback('task_radio', new_value)

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
