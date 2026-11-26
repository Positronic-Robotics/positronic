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
}

// The grip pose, the button values and the press counts of the controller of `inputSource` in `frame`, or null when it
// has no pose there. `counter` counts in each frame, so a press in a frame with no pose goes out with the next pose
export function controllerInFrame(inputSource, frame, refSpace, counter) {
  const gamepadButtons = inputSource.gamepad ? inputSource.gamepad.buttons : [];
  const presses = counter.count(gamepadButtons.map(b => b.pressed));
  const gripPose = frame.getPose(inputSource.gripSpace, refSpace);
  if (!gripPose) return null;
  const { position, orientation } = gripPose.transform;
  return { position, orientation, values: gamepadButtons.map(b => b.value), presses };
}
