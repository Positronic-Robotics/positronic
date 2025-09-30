import platform

import configuronic as cfn


@cfn.config()
def sound(**kwargs):
    if platform.system() == 'Darwin':
        return None

    import positronic.drivers.sound

    return positronic.drivers.sound.SoundSystem(**kwargs)
