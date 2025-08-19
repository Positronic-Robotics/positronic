import configuronic as cfn


@cfn.config()
def sound(**kwargs):
    import positronic.drivers.sound
    return positronic.drivers.sound.SoundSystem(**kwargs)
