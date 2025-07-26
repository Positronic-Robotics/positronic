import configuronic as cfn
import pimm.drivers.sound

sound = cfn.Config(
    pimm.drivers.sound.SoundSystem,
)
