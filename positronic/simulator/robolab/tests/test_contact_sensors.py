from dataclasses import dataclass, field

from positronic.simulator.robolab import contact_sensors


@dataclass
class _ContactCfg:
    history_length: int = 6


@dataclass
class _OtherCfg:
    history_length: int = 6


@dataclass
class _Scene:
    gripper__all_objs: _ContactCfg = field(default_factory=_ContactCfg)
    cube__bin: _ContactCfg = field(default_factory=_ContactCfg)
    imu: _OtherCfg = field(default_factory=_OtherCfg)
    num_envs: int = 8


def test_make_lazy_drops_the_history_of_every_contact_sensor():
    scene = _Scene()

    assert contact_sensors.make_lazy(scene, _ContactCfg) == 2

    assert scene.gripper__all_objs.history_length == 0
    assert scene.cube__bin.history_length == 0


def test_make_lazy_keeps_the_history_of_another_sensor_kind():
    scene = _Scene()

    contact_sensors.make_lazy(scene, _ContactCfg)

    assert scene.imu.history_length == 6
