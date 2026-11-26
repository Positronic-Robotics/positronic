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
