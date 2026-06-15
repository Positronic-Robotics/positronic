from collections import deque
from collections.abc import Iterator
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import pimm
from positronic.dataset.edits import EditedDataset
from positronic.dataset.local_dataset import LocalDataset
from positronic.policy.harness import Directive


class State(Enum):
    WAITING = auto()
    RUNNING = auto()


UNIFIED_TASK = 'Pick all the items one by one from transparent tote and place them into the large grey tote.'

OBJECTS = ['Towels', 'Wooden spoons', 'Scissors', 'Batteries', 'USB-C Boxes', 'USB-C Cables']

TASKS = [
    UNIFIED_TASK,
    'Pick all the towels one by one from transparent tote and place them into the large grey tote.',
    'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.',
    'Pick all the scissors one by one from transparent tote and place them into the large grey tote.',
    'Pick up the green cube and place it on the red cube.',
    'Pick up objects from the red tote and place them in the green tote.',
]

TASK_TO_OBJECT = {TASKS[1]: 'Towels', TASKS[2]: 'Wooden spoons', TASKS[3]: 'Scissors'}

OUTCOMES = ['Success', 'Fail', 'Stalled', 'Ran out of time', 'Safety', 'System']

SIDES = ['left', 'right', 'NA']

EDITOR_POLL_SEC = 0.5


# TODO: split this into a reusable review surface (episode navigation + per-field edits over the dataset's edit log)
# and the lifecycle driver (RUN/FINISH/HOME + time budget). The review half is the sharable one — ideally the
# positronic server grows into it — but the editable static fields differ per eval, so the form must become
# data-driven (declared by the Task/Eval) instead of phail-specific. Design pass needed before factoring.
class EvalUI(pimm.ControlSystem):
    """Operator console for attended evals: trial control plus an episode editor.

    Trial control drives the lifecycle (RUN/FINISH/HOME directives) and shows the remaining time budget.
    The editor is a non-modal view over the recorded dataset: the operator navigates episodes and corrects
    everything except the task itself (outcome, notes, item counts, object, tote/camera placement), with
    each change committed to the edit log as it is made — the stop press itself persists the initial
    annotation. Episodes can be dropped and undropped. New episodes are discovered by polling the dataset
    directory, so reviewing never blocks the next trial.
    """

    def __init__(
        self, output_dir: Path | None = None, max_im_size: tuple[int, int] = (320, 240), ui_scale: float = 1.0
    ):
        self.state = State.WAITING
        self.output_dir = output_dir
        self.ui_scale = ui_scale
        self.max_im_size = (self.size(max_im_size[0]), self.size(max_im_size[1]))

        # --- Inputs/Outputs ---
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.directive = pimm.ControlSystemEmitter(self)

        # UI State
        self.element_states: dict[str | int, list[State]] = {}

        # Internal state for camera rendering
        self.im_sizes = {}
        self.raw_textures = {}

        self.cap_per_item = 30
        self.run_start_time = None

        # Editor state. The editor navigates ALL recorded episodes by raw position (`_base`), showing dropped ones
        # greyed-out rather than filtered away, while `_edited` is the curated/edit handle whose methods amend the log.
        self._base = None
        self._edited: EditedDataset | None = None
        self._count = 0
        self._dropped: frozenset[str] = frozenset()
        self._sel: int | None = None
        self._last_scan = float('-inf')
        # Stop verdicts awaiting their episode: each stop queues one, persisted when its finished episode appears
        # in the dataset directory (FIFO — episodes finalize in stop order). Held no matter what runs in between, so
        # rapid stops, Start, and Reset can't drop a verdict before its recording lands.
        self._pending_reviews: deque[dict] = deque()
        # The selected episode's video signal under the review scrubber.
        self._rv_signal = None
        self.rv_texture = np.zeros((240, 320, 4), dtype=np.float32)

    def size(self, v: int) -> int:
        """Scale a value by ui_scale."""
        return int(v * self.ui_scale)

    def _register(self, tag: str | int, enabled_states: list[State]) -> str | int:
        """Registers a UI element's enabled states."""
        self.element_states[tag] = enabled_states
        return tag

    def _create_theme(self):
        text_cols = (dpg.mvThemeCol_Text, dpg.mvThemeCol_TextDisabled, dpg.mvThemeCol_CheckMark)
        frame_cols = (
            dpg.mvThemeCol_Button,
            dpg.mvThemeCol_ButtonHovered,
            dpg.mvThemeCol_ButtonActive,
            dpg.mvThemeCol_FrameBg,
            dpg.mvThemeCol_FrameBgHovered,
            dpg.mvThemeCol_FrameBgActive,
        )
        with dpg.theme(tag='disabled_theme'):
            for enabled_state in (True, False):
                with dpg.theme_component(dpg.mvAll, enabled_state=enabled_state):
                    for col in text_cols:
                        dpg.add_theme_color(col, (100, 100, 100))
                    for col in frame_cols:
                        dpg.add_theme_color(col, (20, 20, 20))

        button_themes = {
            'finished_theme': ((0, 100, 0), (0, 120, 0), (0, 80, 0)),
            'fail_theme': ((170, 60, 0), (190, 70, 0), (150, 55, 0)),
            'safety_theme': ((150, 0, 0), (170, 0, 0), (130, 0, 0)),
            'system_theme': ((80, 80, 80), (100, 100, 100), (60, 60, 60)),
        }
        for tag, (button, hovered, active) in button_themes.items():
            with dpg.theme(tag=tag), dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, button)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hovered)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, active)

    # Stop buttons: each ends the trial with its outcome as the initial annotation.
    STOP_BUTTONS = [
        ('Finished', 'Success', 'finished_theme'),
        ('Fail', 'Fail', 'fail_theme'),
        ('Safety', 'Safety', 'safety_theme'),
        ('System', 'System', 'system_theme'),
    ]

    def _build_controls(self):
        with dpg.group(horizontal=True):
            self._register(
                dpg.add_button(label='Start', callback=self.start, width=self.size(80), height=self.size(32)),
                [State.WAITING],
            )
            dpg.add_spacer(width=self.size(15))
            for label, reason, theme in self.STOP_BUTTONS:
                btn = dpg.add_button(
                    label=label,
                    callback=self._on_stop_button,
                    user_data=reason,
                    width=self.size(80),
                    height=self.size(32),
                )
                dpg.bind_item_theme(btn, theme)
                self._register(btn, [State.RUNNING])
                dpg.add_spacer(width=self.size(5))
            dpg.add_spacer(width=self.size(5))
            self._register(
                dpg.add_button(label='Reset', callback=self.reset, width=self.size(80), height=self.size(32)),
                [State.WAITING, State.RUNNING],
            )
            dpg.add_spacer(width=self.size(25))
            with dpg.drawlist(width=self.size(160), height=self.size(46)):
                dpg.draw_text((0, 0), '0:00', size=self.size(40), tag='time_text')

    def _build_configuration(self):
        dpg.add_text('Configuration')
        with dpg.group(horizontal=True):
            with dpg.child_window(height=self.size(210), width=self.size(480), border=True):
                dpg.add_text('Task')
                self._register(
                    dpg.add_radio_button(
                        items=[*TASKS, 'Other'], default_value=TASKS[0], callback=self.radio_callback, tag='task_radio'
                    ),
                    [State.WAITING],
                )
                self._register(
                    dpg.add_input_text(show=False, width=self.size(350), tag='custom_input'), [State.WAITING]
                )

            dpg.add_spacer(width=self.size(5))

            with dpg.child_window(height=self.size(210), width=self.size(200), border=True, tag='object_window'):
                dpg.add_text('Object', tag='object_label')
                self._register(
                    dpg.add_radio_button(
                        items=[*OBJECTS, 'Other'],
                        default_value=OBJECTS[0],
                        callback=self.radio_callback,
                        tag='object_radio',
                    ),
                    [State.WAITING],
                )
                self._register(
                    dpg.add_input_text(show=False, width=self.size(180), tag='object_custom_input'), [State.WAITING]
                )

        dpg.add_spacer(height=self.size(25))
        with dpg.group(horizontal=True):
            self._register(
                dpg.add_input_int(
                    label='Total items',
                    default_value=1,
                    step=1,
                    min_value=1,
                    min_clamped=True,
                    width=self.size(80),
                    tag='total_items_input',
                    callback=self.validate_items_callback,
                ),
                [State.WAITING],
            )
            dpg.add_spacer(width=self.size(20))
            self._register(
                dpg.add_input_int(
                    label='Successful',
                    default_value=0,
                    step=1,
                    min_value=0,
                    min_clamped=True,
                    width=self.size(80),
                    tag='successful_items_input',
                    callback=self.validate_items_callback,
                ),
                [State.RUNNING],
            )
            dpg.add_spacer(width=self.size(30))
            self._register(
                dpg.add_input_int(
                    label='Cap/item (s)',
                    default_value=self.cap_per_item,
                    width=self.size(80),
                    tag='cap_per_item_input',
                    callback=self.cap_callback,
                    step=1,
                ),
                [State.WAITING],
            )

        dpg.add_spacer(height=self.size(20))
        with dpg.group(horizontal=True):
            dpg.add_text('Tote Placement')
            self._register(
                dpg.add_radio_button(items=SIDES, default_value='NA', horizontal=True, tag='tote_radio'),
                [State.WAITING],
            )
            dpg.add_spacer(width=self.size(20))
            dpg.add_text('External Camera')
            self._register(
                dpg.add_radio_button(items=SIDES, default_value='NA', horizontal=True, tag='camera_radio'),
                [State.WAITING],
            )

    # Editor fields that commit when the operator leaves the widget after editing; radios commit on click.
    TEXT_FIELDS = {
        'ed_total': 'eval.total_items',
        'ed_success': 'eval.successful_items',
        'ed_notes': 'eval.notes',
        'ed_object': 'eval.object',
    }
    RADIO_FIELDS = {'ed_outcome': 'eval.outcome', 'ed_tote': 'eval.tote_placement', 'ed_camera': 'eval.external_camera'}

    def _build_editor(self):
        with dpg.group(horizontal=True):
            dpg.add_button(label='<', tag='ed_prev', callback=lambda: self._select(self._sel - 1), width=self.size(28))
            dpg.add_text('-/-', tag='ed_pos')
            dpg.add_button(label='>', tag='ed_next', callback=lambda: self._select(self._sel + 1), width=self.size(28))
            dpg.add_spacer(width=self.size(15))
            dpg.add_text('', tag='ed_time')
            dpg.add_spacer(width=self.size(15))
            dpg.add_text('', tag='ed_status', color=(220, 60, 60))
        dpg.add_spacer(height=self.size(5))
        dpg.add_text('', tag='ed_task', wrap=self.size(560))
        dpg.add_spacer(height=self.size(5))
        with dpg.group(horizontal=True):
            dpg.add_text('Outcome')
            dpg.add_spacer(width=self.size(5))
            dpg.add_radio_button(
                items=OUTCOMES,
                default_value=OUTCOMES[0],
                horizontal=True,
                tag='ed_outcome',
                callback=self._commit_field,
                user_data='ed_outcome',
            )
        dpg.add_spacer(height=self.size(5))
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label='Total items', step=1, min_value=1, min_clamped=True, width=self.size(80), tag='ed_total'
            )
            dpg.add_spacer(width=self.size(20))
            dpg.add_input_int(
                label='Successful', step=1, min_value=0, min_clamped=True, width=self.size(80), tag='ed_success'
            )
            dpg.add_spacer(width=self.size(20))
            dpg.add_input_text(label='Object', width=self.size(160), tag='ed_object')
        dpg.add_spacer(height=self.size(5))
        with dpg.group(horizontal=True):
            dpg.add_text('Tote')
            dpg.add_radio_button(
                items=SIDES,
                default_value='NA',
                horizontal=True,
                tag='ed_tote',
                callback=self._commit_field,
                user_data='ed_tote',
            )
            dpg.add_spacer(width=self.size(20))
            dpg.add_text('Camera')
            dpg.add_radio_button(
                items=SIDES,
                default_value='NA',
                horizontal=True,
                tag='ed_camera',
                callback=self._commit_field,
                user_data='ed_camera',
            )
        dpg.add_spacer(height=self.size(5))
        with dpg.group(horizontal=True):
            dpg.add_input_text(multiline=True, height=self.size(60), width=self.size(380), tag='ed_notes')
            dpg.add_spacer(width=self.size(10))
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label='Drop', tag='ed_drop', width=self.size(100), height=self.size(28), callback=self.drop_episode
                )
                dpg.add_button(
                    label='Undrop',
                    tag='ed_undrop',
                    width=self.size(100),
                    height=self.size(28),
                    callback=self.undrop_episode,
                    show=False,
                )
        dpg.add_spacer(height=self.size(10))
        with dpg.group(horizontal=True):
            dpg.add_text('Recorded video')
            dpg.add_spacer(width=self.size(15))
            dpg.add_combo(items=[], tag='rv_camera', width=self.size(200), callback=self._on_review_camera)
        dpg.add_image('rv_tex', width=self.size(320), height=self.size(240))
        with dpg.group(horizontal=True):
            dpg.add_slider_int(tag='rv_slider', width=self.size(250), callback=lambda s, a: self._show_frame(a))
            dpg.add_spacer(width=self.size(10))
            dpg.add_text('', tag='rv_time')
        for tag in self.TEXT_FIELDS:
            with dpg.item_handler_registry() as reg:
                dpg.add_item_deactivated_after_edit_handler(callback=self._commit_field, user_data=tag)
            dpg.bind_item_handler_registry(tag, reg)

    def _setup_key_handlers(self):
        with dpg.handler_registry():

            def safe_trigger(callback):
                text_inputs = [
                    'custom_input',
                    'object_custom_input',
                    'total_items_input',
                    'successful_items_input',
                    'cap_per_item_input',
                    *self.TEXT_FIELDS,
                ]
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

    # --- State Transitions ---

    def start(self, sender=None, app_data=None):
        if self.state != State.WAITING:
            return
        print('State: RUNNING')
        self.state = State.RUNNING
        self.update_ui()

        task_value = dpg.get_value('task_radio')
        task_name = dpg.get_value('custom_input') if task_value == 'Other' else task_value
        context = {'task': task_name}

        if task_value in TASK_TO_OBJECT or task_value == UNIFIED_TASK:
            obj_value = dpg.get_value('object_radio')
            if obj_value == 'Other':
                obj_value = dpg.get_value('object_custom_input')
            context['eval.object'] = obj_value

        tote_val = dpg.get_value('tote_radio')
        if tote_val != 'NA':
            context['eval.tote_placement'] = tote_val
        camera_val = dpg.get_value('camera_radio')
        if camera_val != 'NA':
            context['eval.external_camera'] = camera_val
        context['eval.total_items'] = dpg.get_value('total_items_input')
        context['eval.cap_per_item'] = self.cap_per_item

        self.run_start_time = self.clock.now()
        self.directive.emit(Directive.RUN(**context))

    # Parametrized dpg callbacks take the (sender, app_data, user_data) form with the parameter riding
    # ``user_data``: dpg builds the argument list from the callback's arity and cannot introspect
    # ``functools.partial`` objects, so a partial-bound callback never fires.
    def _on_stop_button(self, sender, app_data, user_data):
        self.stop_run(user_data)

    def stop_run(self, reason):
        if self.state != State.RUNNING:
            return
        print(f'State: WAITING ({reason})')
        self.state = State.WAITING
        total = dpg.get_value('total_items_input')
        successful = total if reason == 'Success' else dpg.get_value('successful_items_input')
        self._pending_reviews.append({'outcome': reason, 'successful': successful})

        self.update_ui()
        self.directive.emit(Directive.FINISH())

    def reset(self, sender=None, app_data=None):
        print('State: WAITING (Reset)')
        self.state = State.WAITING
        self.update_ui()
        self.directive.emit(Directive.HOME())

    # --- Episode editor ---

    def _refresh_view(self):
        """Rebuild the editor over the dataset directory; call after new episodes land on disk."""
        self._base = LocalDataset(self.output_dir)
        self._edited = EditedDataset(self._base, self.output_dir)
        self._dropped = self._edited.dropped
        self._count = len(self._base)

    def _select(self, idx: int):
        """Select an episode and populate the review form from its current (edits-applied) state."""
        if self._count == 0:
            self._sel = None
        else:
            self._sel = max(0, min(idx, self._count - 1))
            ep = self._edited.overlay(self._base[self._sel])
            static = ep.static
            dpg.set_value('ed_task', static.get('task', ''))
            dpg.set_value('ed_outcome', static.get('eval.outcome', OUTCOMES[0]))
            dpg.set_value('ed_total', static.get('eval.total_items', 1))
            dpg.set_value('ed_success', static.get('eval.successful_items', 0))
            dpg.set_value('ed_notes', static.get('eval.notes', ''))
            dpg.set_value('ed_object', static.get('eval.object', ''))
            dpg.set_value('ed_tote', static.get('eval.tote_placement', 'NA'))
            dpg.set_value('ed_camera', static.get('eval.external_camera', 'NA'))
            recorded_at = datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9)
            dpg.set_value('ed_time', recorded_at.strftime('%Y-%m-%d %H:%M:%S'))
        self._bind_review_video()
        self._update_editor_ui()

    def _poll_episodes(self):
        """Pick up newly finished episodes; the recorder's `.unfinished` marker hides in-progress ones."""
        if self.output_dir is None or len(LocalDataset(self.output_dir)) == self._count:
            return
        prev_count = self._count
        self._refresh_view()
        # Each newly landed episode claims the next queued stop verdict (its initial annotation) and is persisted
        # right away, so corrections start from the operator's call.
        seeded = False
        for idx in range(prev_count, self._count):
            if not self._pending_reviews:
                break
            review = self._pending_reviews.popleft()
            uid = self._base[idx].meta['uid']
            self._edited = self._edited.set_static(
                uid, {'eval.outcome': review['outcome'], 'eval.successful_items': review['successful']}
            )
            seeded = True
        if seeded:
            dpg.set_value('mode_tabs', 'tab_episodes')
        self._select(self._count - 1)

    def _commit_field(self, sender, app_data, user_data):
        if user_data in ('ed_total', 'ed_success'):
            # Same invariant the live controls enforce: successful never exceeds total.
            total = dpg.get_value('ed_total')
            successful = min(dpg.get_value('ed_success'), total)
            dpg.set_value('ed_success', successful)
            data = {'eval.total_items': total, 'eval.successful_items': successful}
        else:
            key = (self.TEXT_FIELDS | self.RADIO_FIELDS)[user_data]
            data = {key: dpg.get_value(user_data)}
        self._edited = self._edited.set_static(self._base[self._sel].meta['uid'], data)

    def _bind_review_video(self):
        # GUI video binding follows the `image.` observation naming convention, like the live feeds.
        ep = self._base[self._sel] if self._sel is not None else None
        cams = sorted(k for k in ep if k.startswith('image.')) if ep is not None else []
        dpg.configure_item('rv_camera', items=cams)
        if not cams:
            self._rv_signal = None
            dpg.set_value('rv_time', '')
            self.rv_texture[:] = 0.0
            return
        cam = dpg.get_value('rv_camera')
        if cam not in cams:
            cam = cams[0]
            dpg.set_value('rv_camera', cam)
        self._rv_signal = ep[cam]
        last = len(self._rv_signal) - 1
        dpg.configure_item('rv_slider', max_value=last)
        dpg.set_value('rv_slider', last)
        self._show_frame(last)

    def _on_review_camera(self, sender=None, app_data=None):
        self._rv_signal = self._base[self._sel][app_data]
        last = len(self._rv_signal) - 1
        dpg.configure_item('rv_slider', max_value=last)
        idx = min(dpg.get_value('rv_slider'), last)
        dpg.set_value('rv_slider', idx)
        self._show_frame(idx)

    def _show_frame(self, idx: int):
        if self._rv_signal is None:
            return
        idx = max(0, min(int(idx), len(self._rv_signal) - 1))
        frame, ts = self._rv_signal[idx]
        h, w = frame.shape[:2]
        th, tw = self.rv_texture.shape[:2]
        scale = min(tw / w, th / h)
        dw, dh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        self.rv_texture[:] = 0.0
        y0, x0 = (th - dh) // 2, (tw - dw) // 2
        self.rv_texture[y0 : y0 + dh, x0 : x0 + dw, :3] = resized / 255.0
        self.rv_texture[y0 : y0 + dh, x0 : x0 + dw, 3] = 1.0
        t0 = self._rv_signal.start_ts
        dpg.set_value('rv_time', f'{(ts - t0) / 1e9:.1f} / {(self._rv_signal.last_ts - t0) / 1e9:.1f}s')

    def drop_episode(self, sender=None, app_data=None):
        self._edited = self._edited.drop(self._base[self._sel].meta['uid'])
        self._dropped = self._edited.dropped
        self._select(self._sel)

    def undrop_episode(self, sender=None, app_data=None):
        self._edited = self._edited.undrop(self._base[self._sel].meta['uid'])
        self._dropped = self._edited.dropped
        self._select(self._sel)

    def _set_enabled(self, tag, enabled: bool):
        dpg.bind_item_theme(tag, 0 if enabled else 'disabled_theme')
        dpg.configure_item(tag, enabled=enabled)

    def _update_editor_ui(self):
        if self._sel is None:
            dpg.set_value('ed_pos', '-/-')
            dpg.set_value('ed_time', '')
            dpg.set_value('ed_status', '')
            dropped = False
        else:
            dropped = self._base[self._sel].meta['uid'] in self._dropped
            dpg.set_value('ed_pos', f'{self._sel + 1}/{self._count}')
            dpg.set_value('ed_status', 'DROPPED' if dropped else '')
        editable = self._sel is not None and not dropped
        for tag in (*self.TEXT_FIELDS, *self.RADIO_FIELDS):
            self._set_enabled(tag, editable)
        self._set_enabled('ed_prev', self._sel is not None and self._sel > 0)
        self._set_enabled('ed_next', self._sel is not None and self._sel < self._count - 1)
        dpg.configure_item('ed_drop', show=not dropped)
        self._set_enabled('ed_drop', self._sel is not None and not dropped)
        dpg.configure_item('ed_undrop', show=dropped)

    # --- Callbacks ---

    def radio_callback(self, sender, app_data):
        # Guard: Only allow change in WAITING
        if self.state != State.WAITING:
            return

        # Only pass task_value when the task radio itself changed
        self.update_ui(task_value=app_data if sender == 'task_radio' else None)

    def cap_callback(self, sender, app_data):
        self.cap_per_item = app_data
        self.update_ui()

    def validate_items_callback(self, sender, app_data):
        total = dpg.get_value('total_items_input')
        successful = dpg.get_value('successful_items_input')

        if total < 1:
            total = 1
            dpg.set_value('total_items_input', total)

        if successful < 0:
            successful = 0
            dpg.set_value('successful_items_input', successful)

        if successful > total:
            dpg.set_value('successful_items_input', total)

        self.update_ui()

    # --- UI Update ---

    def _set_timer(self, seconds: float):
        m, s = divmod(max(0, int(seconds)), 60)
        dpg.configure_item('time_text', text=f'{m}:{s:02d}')

    def update_ui(self, task_value=None):
        for tag, enabled_states in self.element_states.items():
            self._set_enabled(tag, self.state in enabled_states)

        # Special logic for dependent widgets
        if task_value is None:
            task_value = dpg.get_value('task_radio')

        is_other = task_value == 'Other'
        dpg.configure_item('custom_input', show=is_other)

        # Object window visibility
        if task_value == UNIFIED_TASK:
            dpg.configure_item('object_window', show=True)
            dpg.configure_item('object_radio', enabled=self.state == State.WAITING)
            dpg.configure_item('object_custom_input', show=dpg.get_value('object_radio') == 'Other')
        elif task_value in TASK_TO_OBJECT:
            dpg.configure_item('object_window', show=True)
            dpg.configure_item('object_radio', enabled=False)
            dpg.set_value('object_radio', TASK_TO_OBJECT[task_value])
            dpg.configure_item('object_custom_input', show=False)
        else:
            dpg.configure_item('object_window', show=False)

        if self.state == State.WAITING:
            dpg.set_value('successful_items_input', 0)
            self._set_timer(dpg.get_value('total_items_input') * self.cap_per_item)

    # --- Control System Run Loop ---

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self.clock = clock
        # Initialize DPG Context
        dpg.create_context()
        self._create_theme()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(320, 240, default_value=self.rv_texture, format=dpg.mvFormat_Float_rgba, tag='rv_tex')

        # Window
        with dpg.window(label='Evaluation Control', width=self.size(1200), height=self.size(800), tag='main_window'):
            with dpg.group(horizontal=True):
                # Left side: live camera feeds, fixed in place across tabs so the operator's view never jumps.
                with dpg.group(horizontal=False, tag='image_grid_group'):
                    dpg.add_text('Camera Feed')

                # Spacer between images and controls
                dpg.add_spacer(width=self.size(20))

                # Right side: Controls
                with dpg.group(horizontal=False):
                    dpg.add_text('Controls')
                    dpg.add_spacer(height=self.size(5))
                    self._build_controls()

                    dpg.add_spacer(height=self.size(10))
                    with dpg.tab_bar(tag='mode_tabs'):
                        with dpg.tab(label='Trial', tag='tab_trial'):
                            dpg.add_spacer(height=self.size(10))
                            self._build_configuration()
                        with dpg.tab(label='Episodes', tag='tab_episodes'):
                            dpg.add_spacer(height=self.size(10))
                            self._build_editor()

        self._setup_key_handlers()

        dpg.create_viewport(title='Eval UI', width=self.size(520), height=self.size(850))
        dpg.set_viewport_vsync(True)
        dpg.configure_app(keyboard_navigation=True)
        if self.ui_scale != 1.0:
            dpg.set_global_font_scale(self.ui_scale)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()

        # Initialize UI state; open the editor on the newest episode of an existing dataset
        self.update_ui()
        if self.output_dir is not None:
            self._refresh_view()
        self._select(self._count - 1)

        while not should_stop.value and dpg.is_dearpygui_running():
            now = self.clock.now()
            if now - self._last_scan >= EDITOR_POLL_SEC:
                self._last_scan = now
                self._poll_episodes()

            # Count down the remaining time budget; the run stops when it is exhausted
            if self.state == State.RUNNING and self.run_start_time:
                elapsed = now - self.run_start_time
                total_cap = dpg.get_value('total_items_input') * self.cap_per_item
                self._set_timer(total_cap - elapsed)
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
            yield pimm.Yield()

        dpg.destroy_context()


def main():
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

        world.run([ui, fake_camera])


if __name__ == '__main__':
    main()
