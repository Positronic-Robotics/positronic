# Nebius competition

Walkthroughs for the competition's sim qualifier, on top of the same client every other caller uses.

- `submit_sample.py` — submit a policy image to the `robolab.public_subset` eval, wait for it to
  finish, and report what it scored.

Three things about the competition flow that the generic walkthrough does not cover:

- **Pin your image by digest.** The platform resolves the reference to a digest at submission time
  and runs those exact bytes; submitting a mutable tag means the run may not be the build you
  tested.
- **Reuse a transaction key on a retry.** A retry with the same key returns the original
  submission; a retry without one spends another day's quota. The key is yours to choose, and
  reusing it with a *different* request is a conflict rather than a silent swap.
- **A board is per eval version.** A rotation changes the version and starts a new board, so scores
  from before it are never mixed with scores from after.
