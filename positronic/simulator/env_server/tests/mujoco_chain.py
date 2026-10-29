"""A small MuJoCo arm for tests: a chain of hinges with a control site at its tip."""

_AXES = ('1 0 0', '0 0 1', '0 1 0')


def hinge_chain_xml(arm: str, y: float, joints: int) -> str:
    """Body ``{arm}_base`` at ``y``, hinges ``{arm}_joint1`` to ``{arm}_joint{joints}``, site ``{arm}_grasp_site``."""
    links = ''.join(
        f'<body name="{arm}_link{i}" pos="0 0 0.1"><joint name="{arm}_joint{i}" axis="{_AXES[i % 3]}" '
        'range="-3 3"/><geom size="0.02" mass="0.1"/>'
        for i in range(1, joints + 1)
    )
    tip = f'<site name="{arm}_grasp_site" pos="0 0 0.1"/>'
    return f'<body name="{arm}_base" pos="0 {y} 0">{links}{tip}{"</body>" * joints}</body>'
