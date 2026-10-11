# positronic-eval-vocabulary

The words an attended rollout writes into an episode, and what they mean.

An operator scores an episode and marks how far the arm got. Both reach the recording as strings —
`eval.outcome` and the `progress.state` signal — and several programs then read them back: the
console that wrote them, the viewer that colours them, a report that counts them, a coordinator that
has no robot installed at all.

An episode keeps its statics in one file and its id in another. A labeler reads from them how the
trial ended, how long it ran, who ended it, the task and the id. `eval_vocabulary.episode` names the
files, the keys, and the signal file names that an older recording takes its end from.

This distribution defines those words once. It depends on nothing, so any of those
programs installs it alone.

```python
from eval_vocabulary.outcome import ABSENT, OUTCOME, is_scored

outcome = episode.get(OUTCOME, ABSENT)
scored = is_scored(outcome)
```

A reader is tolerant and a writer is strict. A program that writes a verdict constructs `Outcome`,
so a word outside this vocabulary raises where it is written. A program that reads one takes the
string as it finds it, because a recording may predate a word this copy knows.
