# Code Rules

Judgment rules for this repository — the checks a linter cannot make. Read by the Codex reviewer on
pull requests, by Claude while implementing, and by the `check-rules` skill.

Mechanical checks live in `pyproject.toml` (ruff, basedpyright), never here. A rule a linter could
enforce is a rule that would be applied inconsistently.

Every rule has an **id** — its heading. Cite it when reporting, fixing, or waiving a violation. Add,
calibrate, and retire rules through the `add-rule` skill.

## Waiving a rule

A violation is waived when the offending line or its enclosing block carries:

```python
# rules-allow: <rule-id> — <reason>
```

The reason is required, and must say why this instance is correct — not that the rule is inconvenient.
A waiver is as narrow as a `noqa`: one rule id, at the site. To retire a rule everywhere, delete it
from this file.

## Rules

### caller-in-name

Don't name anything — function, class, module, variable — after where it is used. Name it after what it
does. Information about the callers must not leak into the name.

A helper that recomposes a pose through a fixed transform is `change_frame`, not `to_policy_frame`: it
has never heard of a policy.

### hardcoded-keys

Don't write the names of data fields as literals inside your code — dict keys, signal names, field
paths. Take them as parameters instead, with a default for the usual name.

Exception: a name the component itself owns and defines, rather than one it reads from its input.

```python
# Bad — only works on data that spells these three names exactly this way
class ChangeEEFrame:
    def encode(self, data):
        transform = frame_transform(data['urdf'], data['control_frame'], self._to)
        return {**data, 'robot_state.ee_pose': change_frame(data['robot_state.ee_pose'], transform)}


# Good — a second dataset passes urdf_key='robot/urdf' and reuses the class
class ChangeEEFrame:
    def __init__(self, to, pose_key='robot_state.ee_pose', urdf_key='urdf', control_frame_key='control_frame'):
        ...

    def encode(self, data):
        transform = frame_transform(data[self._urdf_key], data[self._control_frame_key], self._to)
        return {**data, self._pose_key: change_frame(data[self._pose_key], transform)}
```
