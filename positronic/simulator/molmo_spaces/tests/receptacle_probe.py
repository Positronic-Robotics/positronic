"""Print what ``ObjectManager.objects_on_receptacle`` finds in small scenes, and the bodies whose box it computes.

Runs in the MolmoSpaces interpreter with the launcher's subprocess environment. Prints one JSON object that maps
each scene name to the found object names and the boxed body names, in call order.
"""

import json
from typing import Any
from unittest import mock

# env.py installs the CGL stub before loading MolmoSpaces; import it before any other molmo_spaces import.
import env
import mujoco

# MolmoSpaces is installed only in the server's isolated environment.
from molmo_spaces.env.object_manager import ObjectManager  # pyright: ignore[reportMissingImports]
from molmo_spaces.utils import mujoco_scene_utils  # pyright: ignore[reportMissingImports]

RECEPTACLE = 'receptacle'
PICKUP = 'pickup'

# A receptacle of two boxes on the floor, and a box to put on it. Both bodies are free.
_SCENE_XML = f"""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="{RECEPTACLE}">
      <freejoint/>
      <geom type="box" size="0.05 0.1 0.01" pos="-0.05 0 0.01"/>
      <geom type="box" size="0.05 0.1 0.01" pos="0.05 0 0.01"/>
    </body>
    <body name="{PICKUP}">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""
_RECEPTACLE_TOP = 0.02
_PICKUP_HALF = 0.02
_DEPTH = 0.0005  # The penetration of a resting contact.
_GAP = 0.005  # Less than the 1 cm that the fallback stage accepts.

# The pickup position and whether the scene metadata lists the pickup, which the fallback stage requires.
_SCENES = {
    'on_floor': ((0.5, 0.0, _PICKUP_HALF - _DEPTH), True),
    'held_by_contact': ((0.02, 0.0, _RECEPTACLE_TOP + _PICKUP_HALF - 2 * _DEPTH), False),
    'just_above': ((0.02, 0.0, _RECEPTACLE_TOP + _PICKUP_HALF + _GAP), True),
}


class _Env:
    """The parts of a MolmoSpaces env that an ``ObjectManager`` reads."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict[str, Any]) -> None:
        self.mj_datas = [data]
        self.current_model = model
        self.current_scene_metadata = metadata


def _probe(pickup_pos: tuple[float, float, float], pickup_in_metadata: bool) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    data = mujoco.MjData(model)
    receptacle_qpos = model.jnt_qposadr[model.body(RECEPTACLE).jntadr[0]]
    pickup_qpos = model.jnt_qposadr[model.body(PICKUP).jntadr[0]]
    data.qpos[receptacle_qpos : receptacle_qpos + 3] = (0.0, 0.0, -_DEPTH)
    data.qpos[pickup_qpos : pickup_qpos + 3] = pickup_pos
    mujoco.mj_forward(model, data)

    metadata = {'objects': {PICKUP: {}} if pickup_in_metadata else {}}
    om = ObjectManager(_Env(model, data, metadata), 0)
    boxed: list[str] = []
    body_aabb = mujoco_scene_utils.body_aabb

    def counted_body_aabb(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, *args: Any) -> Any:
        boxed.append(model.body(int(body_id)).name)
        return body_aabb(model, data, body_id, *args)

    with mock.patch.object(mujoco_scene_utils, 'body_aabb', counted_body_aabb):
        found = om.objects_on_receptacle([om.get_object_by_name(PICKUP)], om.get_object_by_name(RECEPTACLE).geom_ids)
    return {'found': [obj.name for obj in found], 'boxed': boxed, 'contacts': int(data.ncon)}


def main() -> None:
    env.use_fast_objects_on_receptacle()
    print(json.dumps({name: _probe(*scene) for name, scene in _SCENES.items()}))


if __name__ == '__main__':
    main()
