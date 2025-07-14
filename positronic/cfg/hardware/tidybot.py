import configuronic as cfgc


@cfgc.config(max_vel=(1.0, 1.0, 3.14),
             max_accel=(0.5, 0.5, 2.36),
             encoder_magnet_offsets=[-1675.0 / 4096, -1483.0 / 4096, 1055.0 / 4096, 438.0 / 4096])
def vehicle0(max_vel, max_accel, encoder_magnet_offsets):
    from positronic.drivers.tidybot2 import Tidybot
    return Tidybot(max_vel=max_vel, max_accel=max_accel, encoder_magnet_offsets=encoder_magnet_offsets)
