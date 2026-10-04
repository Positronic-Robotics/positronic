# Policy journals

A journal records one episode of a policy: its startup, each turn and its close. A turn is one call of
the policy. An offline replay reruns a fresh policy against the journal. It starts no world, prepares
no robot and runs no recorded activity.

Run the two examples:

```bash
uv run --locked positronic/cli/examples/policy_journal/record.py /tmp/journal/rollout
uv run --locked positronic/cli/examples/policy_journal/replay.py /tmp/journal/rollout /tmp/journal/branches
```

`record.py` journals a simulated motor through a real `World`, and the inference runs. `replay.py`
calls `verify` with an inference that raises if it runs. Then it branches the journal at the first
inference: `ReplaceResult` supplies a result, and `RerunActivity(..., allow_execution=True)` runs a
second inference on the retained input. Each branch publishes at the recorded turn and time. A branch
whose policy asks for a step plan version that the source does not record stops with `MissingResult`.
The source journal does not change.

## Journal replay and command playback

[`replay_record.py`](../../replay_record.py) loads the simulator state of a dataset episode and plays its
recorded commands at their recorded times through
[`DsPlayerAgent`](../../dataset/ds_player_agent.py). It runs no policy, so it cannot show a policy decision.
[`verify`](../replay.py) is the opposite: it gives a fresh policy the recorded observations, times and
published activity outcomes, and compares what the policy decides. It needs no world, no hardware and no
activity.

## Record

Declare submitted work as `Activity(operation, version, function)`. Increase the version when the same
input can give a different result. Add `journal=Journal(path)` to the `Rollout`.

- At startup, inside a turn and at close, `runtime.time_ns` and every `Answer` stay the same.
  `runtime.invocation` is -1 at startup. A completion is published at the start of the next turn, in
  submission order. That turn can be at the same clock time. An observation that its codec refuses
  starts no turn.
- `runtime.submit` takes only an `Activity`, at startup or inside a turn. The work gets its own decoded
  copy of the arguments. The policy gets observations and results that it cannot change: read-only
  mappings, tuples and read-only arrays. A domain type needs a `PayloadCodec` whose `decode_frozen`
  returns a value that no code can change.
- A failed activity raises `ActivityFailed` with the recorded message and no cause, live and in replay.
  So a fallback that the policy chooses replays too. The journal keeps the traceback. Code that catches
  the original exception type must catch `ActivityFailed` instead.
- [`Codec.wrap`](../codec.py) of an `Activity`, as in `Sequential(ChunkedSchedule(...), codec)`, gives an
  `Activity` whose input is the observation and whose result is the decoded output. Its operation adds
  the codec's wire spec. This needs a codec with `to_spec` that gives plain JSON, and an `Activity` with
  the default `PlainData`; otherwise `wrap` raises `TypeError` when the stack starts. For another payload
  codec, declare the `Activity` yourself around `codec.wrap(function)`. A plain callable and a codec
  without `to_spec` compose as usual.
- The runtime keeps a published result in memory only while the policy holds its `Answer`.
- The journal records each command after the harness emits it, and how the episode ended: the terminal
  payload, a stop or the error.

## What a replay reports

| Change | Result |
| --- | --- |
| The journal has another format, turn semantics, observation codec or command codec | `ValueError` when the journal is read |
| The journal's events are out of order | `ValueError` while `verify` or `branch` plans the replay, before the policy starts |
| The policy is of another class | `ReplayError`, before the policy starts |
| `verify`: the policy records a different event, for example its startup outcome, a step, a turn failure, a submission, a read, a cancellation or the metadata at its close | `ReplayDivergence` at the first event that differs, with `expected` and `actual` |
| `verify`: a finalizer raises in the recorded close of a complete journal | The finalizer's error propagates |
| `branch`: the policy submits work that does not match the recorded submission: operation, version, turn, input, payload codec and capture | `MissingResult`; the branch journal ends before that submission |
| `branch`: the policy fails at startup or in a turn where the source did not fail | The branch closes at the time of that scope and ends with `Raised` and that error |
| `branch`: the policy goes on where the source failed | The branch journal has no `Ended` event, and `verify` reports it incomplete |
| `branch`: a rerun without `allow_execution=True`, or of a submission without a retained input | `ExecutionRefused` or `MissingInput`, before the branch writes anything |
| `branch`: a change to an unknown or unpublished submission, or a result that its codec cannot encode | `ReplayError` |
| A new implementation or new weights behind the same operation and version | Not detected |

The journal names an activity only by its operation, version, codecs and input. It does not hash source
code or weights. After a change to the model, increase the version, or branch with `RerunActivity(...,
allow_execution=True)` to run the new function on the retained inputs; if it raises, the policy gets
`ActivityFailed`, as live. `ReplaceResult` puts in a result that you supply. Both are deliberate branches
against the recorded observations and publication times. They do not predict physics or latency.

The terminal payload or stop that ends an episode comes from the harness, and a replay does not compute
it again. A replay or a branch whose startup and turns end as the source's did keeps the recorded close
time and termination. A branch that avoids the source's policy failure has no recorded history after it,
so its journal is incomplete and does not inherit the source episode's termination; no later observations
exist to go on.

## Limits

- Replay is exact only for the same policy code and initial state. Replay cannot detect a read of a
  clock, a random generator, a file or a network outside the runtime.
- Turn times are on the episode clock. The journal records when an outcome became visible to the
  policy. It does not record when the work ran or how long it waited in the queue.
- A branch keeps the recorded observations. It does not show how a robot moves under the changed commands.
- A [`RemotePolicy`](../remote.py) opens its server session and builds its declared processor stack
  outside every activity, so a journal cannot record them. On a journaled runtime, `RemotePolicy.run`
  raises `TypeError` before it opens the session, in any processor, live and in replay. On a runtime
  without a journal it runs as usual. A journal cannot replay a server session.
- The writer does not sync to disk. After a crash, the reader drops a torn last line. `verify` then
  checks the journal through its last closed scope and reports it incomplete. The policy then closes
  outside the record: an error of its finalizers is logged and does not change the result.
