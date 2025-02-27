from hydra_zen import instantiate

import ironic as ir
from cfg.env import umi, store, builds
from cfg.hardware.camera import cam_store

from tests.stub.hardware.camera import camera_stub


def test_umi_env_is_instantiated():
    store.add_to_hydra_store()
    cam_store.add_to_hydra_store()

    env = instantiate(umi)

    assert isinstance(env, ir.ControlSystem)
    assert isinstance(env.outs.frame, ir.OutputPort)
    assert len(env._components) == 2, "umi should have 2 components: UmiCS and Camera"


def test_umi_env_camera_could_be_overridden():
    store.add_to_hydra_store()
    cam_store.add_to_hydra_store()
    env = instantiate(builds(umi, camera=camera_stub))

    assert isinstance(env, ir.ControlSystem)
