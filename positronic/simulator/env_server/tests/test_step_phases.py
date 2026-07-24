import pytest

from positronic.simulator.env_server.server import disjoint_step_phases


def test_disjoint_step_phases_residual_is_the_server_other_wall():
    phases = disjoint_step_phases(1.0, physics_s=0.6, render_s=0.3)
    assert phases == {'physics_s': 0.6, 'render_s': 0.3, 'server_other_s': pytest.approx(0.1)}
    assert sum(phases.values()) == pytest.approx(1.0)


def test_disjoint_step_phases_clamps_subnanosecond_float_dip():
    phases = disjoint_step_phases(0.9, physics_s=0.6, render_s=0.3 + 1e-9)
    assert phases['server_other_s'] == 0.0


def test_disjoint_step_phases_raises_when_a_phase_is_double_counted():
    # physics + render exceed the whole-step wall: a phase was timed inside another (double count).
    with pytest.raises(ValueError, match='double-counted'):
        disjoint_step_phases(0.5, physics_s=0.4, render_s=0.3)
