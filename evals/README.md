# positronic-evals

The definitions of the public evals, the labelers that read a recorded episode, the rules that score
a run, and MolmoSpaces' committed episode lists.

`positronic` runs these evals on your machine, `positronic-platform-client` names them on the wire,
and a service that scores a run reads them. Each of them installs this distribution. It depends on
`pydantic`, `pyarrow` and `positronic-eval-vocabulary`, and on no robotics stack and no client, so a
service installs it alone:

```bash
uv add "positronic-evals @ git+https://github.com/Positronic-Robotics/positronic@<tag or commit>#subdirectory=evals"
```

`positronic_evals.slug` carries an `IntEnum` on the wire as its lowercase name.

## Public evals

`positronic_evals.evals` carries the definitions of the public evals. A definition pins what the
platform runs under the eval's name: the positronic eval config and its arguments, the trials, the
time limit of one run, its scoring, and the positronic commit that the platform runs the config at.
`public_eval(name)` returns one definition, and `PUBLIC_EVALS` names them all. The platform also
offers evals that it does not publish, and `public_eval` refuses their names.

`positronic eval run` with `--policy` takes a public eval's name as `--eval`, and runs that eval on
your machine. The run uses the positronic you installed. It warns when that positronic is not the
commit the definition names, because the two runs can then differ. A local run does not stop at the
time limit.

## Labels and scores

`positronic_evals.labels` reads what happened in a recorded episode. A labeler reads one episode
directory and gives its `Label`: the episode's id and task, its end, and a list of timed events. An
event is a stage that the arm reached, a count of items delivered, a count of subtasks achieved, or
one of the three end events: `Success`, `Fail` and `Safety`. The earliest end event gives the
`outcome`. A label with no end event ran out of time. The label's other views, such as
`stage_seconds` and `placements`, are properties computed from its events. `Label.until(seconds)`
gives the label as it stood at that time.

`positronic_evals.scoring` states what a label is worth. An eval's `Scoring` names its labelers in
order of precedence, and its primary metric. `label(episode, scoring.labelers)` gives a pair: the
first labeler that labels the episode, and its label. It gives None when no labeler labels the
episode. `score(scoring, labels)` takes the label of each episode, or None. For the episode
directories `episodes` of one run:

```python
from positronic_evals.evals import EvalRef, public_eval
from positronic_evals.labels import label
from positronic_evals.scoring import score

scoring = public_eval(EvalRef('molmo.franka_pick_mini')).scoring
picked = [label(episode, scoring.labelers) for episode in episodes]
scores = score(scoring, [None if given is None else given[1] for given in picked])
```

`score` gives `Scores`, which holds three numbers. `primary` is the mean episode score, or the mean
of the task means where the metric sets `per_task`, and a board ranks on it. `episodes` counts every
episode, and `unscored` counts the episodes that no labeler labelled. `primary` does not count those
as failures. No file holds a score: a score is a function of the labels.

`PUBLIC_LABELERS` holds the labelers that this package publishes, `molmo` and `robolab`. A caller with
labelers of its own passes `label` a table that holds them beside `PUBLIC_LABELERS`. The MolmoSpaces
labeler takes the end of a trial from the episode's signal files: the time from the first sample to
the last, on the world clock. That time can exceed the trial by up to one control step.

The RoboLab labeler reads the end of a trial in the same way. Its task is the task id that RoboLab
names, and not the instruction. It also reads the `subtask` signal, which RoboLab records at each
step, and gives a `SubtaskAchieved` event, with RoboLab's score, for each change of the subtasks
completed. A change of the score alone is not an event. A RoboLab eval scores a success 1.0, and any
other episode the score at its last change of the count, or 0 where it completed no subtask. Where
RoboLab grades progress within a subtask, this score does not count that progress:

```python
from positronic_evals.labels import ROBOLAB_LABELER
from positronic_evals.scoring import EpisodeScore, MeanEpisodeScore, Scoring

robolab = Scoring(
    labelers=(ROBOLAB_LABELER,), primary=MeanEpisodeScore(episode_score=EpisodeScore(use_subtask_score=True))
)
```

## MolmoSpaces episode lists

`positronic_evals.molmo_spaces` holds the episode lists that an offline sweep builds, by benchmark
path below the asset packs' `benchmarks` directory. `RIGHT_VIEW_BLOCKED_PATH` lists the episodes where
scene geometry hides the start grasp site from the right shoulder camera. `BROKEN_EPISODES_PATH` lists
the episodes whose scene cannot be built. The MolmoSpaces sim server leaves the episodes of both lists out
of an open selection and refuses a pinned one. It applies the right-view list only when the run serves the
right shoulder camera. No public eval pins a listed episode.

```python
from positronic_evals.molmo_spaces import RIGHT_VIEW_BLOCKED_PATH, read_episode_list

blocked = read_episode_list(RIGHT_VIEW_BLOCKED_PATH)
```

A benchmark that a list does not name is unswept. Rebuild a list when the pinned MolmoSpaces revision
or the asset packs change. A new list goes beside `right_view_blocked.json`, with its path constant in
`positronic_evals.molmo_spaces`, and `read_episode_list` reads it.
