import configuronic as cfn
import positronic.drivers.sound

sound = cfn.Config(
    positronic.drivers.sound.SoundSystem,
)
