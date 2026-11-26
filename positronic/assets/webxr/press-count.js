// The presses of each button of one controller since the page loaded.
export class PressCounter {
  #counts = [];
  #down = [];

  // Count each button that is down in `pressed` and was not down in the previous call, and return all the counts
  count(pressed) {
    this.#counts = pressed.map((down, index) => (this.#counts[index] ?? 0) + (down && !this.#down[index] ? 1 : 0));
    this.#down = [...pressed];
    return [...this.#counts];
  }

  get counts() {
    return [...this.#counts];
  }
}

// Count the presses of each controller of `session` with its counter in `counters`, by hand. A page calls it in each
// XR frame, also in a frame with no pose, so that no press between two poses is lost
export function countPresses(session, counters) {
  for (const inputSource of session.inputSources) {
    const counter = counters[inputSource.handedness];
    if (counter && inputSource.gamepad) counter.count(inputSource.gamepad.buttons.map(b => b.pressed));
  }
}

// The grip pose, the button values and the press counts of `counter` for the controller of `inputSource` in `frame`,
// or null when it has no pose there
export function controllerInFrame(inputSource, frame, refSpace, counter) {
  const gripPose = frame.getPose(inputSource.gripSpace, refSpace);
  if (!gripPose) return null;
  const { position, orientation } = gripPose.transform;
  const values = inputSource.gamepad ? inputSource.gamepad.buttons.map(b => b.value) : [];
  return { position, orientation, values, presses: counter.counts };
}
