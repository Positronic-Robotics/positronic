# positronic-eval-vocabulary

The words an attended rollout writes into an episode, and what they mean.

An operator scores an episode and marks how far the arm got. Both reach the recording as strings —
`eval.outcome` and the `progress.state` signal — and several programs then read them back: the
console that wrote them, the viewer that colours them, a report that counts them, a coordinator that
has no robot installed at all.

An episode keeps its statics in one file and its id in another. A labeler reads from them how the
trial ended, who ended it, the task and the id. An edit log, a third file, holds the edits that came
after the recording, for example a verdict that a console gave after the policy ended the trial. A
labeler applies those edits over the statics. It reads how long the trial ran from the times in the
signal files. The RoboLab labeler also reads the subtasks that the trial completed from the `subtask`
signal. `eval_vocabulary.episode` names the files, the keys, the edit records and that signal. `eval_vocabulary.timeline`
names the timelines whose columns hold those times.

This distribution defines those words once. It depends on nothing, so any of those
programs installs it alone.

A program that counts episodes gives the statics of each episode to `counted_outcome`. It takes the
first fact that decides the outcome, in this order:

1. The verdict in `eval.outcome`. `Discarded` also counts as it is. `UNSCORED` decides nothing.
2. `eval.terminated` is false: the trial ran out of time.
3. The env reported success or failure in `eval.success`.
4. The policy ended the trial with `give_up`: a failure.
5. The policy ended the trial in an env that ends the trial on success (`eval.env_ends_on_success`):
   a failure, because the env saw no success.

An episode that no fact decides is `UNSCORED`. A rate counts only a verdict, so it leaves out
`UNSCORED` and `Discarded`.

```python
from eval_vocabulary.outcome import counted_outcome, is_scored

outcome = counted_outcome(static)
scored = is_scored(outcome)
```

A reader is tolerant and a writer is strict. A program that writes a verdict constructs `Outcome`,
so a word outside this vocabulary raises where it is written. A program that reads one takes the
string as it finds it, because a recording may predate a word this copy knows.
