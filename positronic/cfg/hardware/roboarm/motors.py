import configuronic as cfn

from positronic.drivers.roboarm.generate_urdf import MotorParameters

my_actuator_rmd_x10_p35_100 = cfn.Config(
    MotorParameters,
    radius=0.061,
    height=0.060,
    mass=1.7,
    effort_limit=100.0,
    velocity_limit=2.0,
)


my_actuator_rmd_x8_p6_20 = cfn.Config(
    MotorParameters,
    radius=0.049,
    height=0.037,
    mass=0.78,
    effort_limit=20.0,
    velocity_limit=2.0,
)

my_actuator_rmd_x6_v3 = cfn.Config(
    MotorParameters,
    radius=0.0395,
    height=0.0295,
    mass=0.49,
    effort_limit=40.0,
    velocity_limit=2.0,
)
