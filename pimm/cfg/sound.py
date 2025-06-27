import ironic as ir
import pimm.drivers.sound

sound = ir.Config(
    pimm.drivers.sound.SoundSystem,
)
