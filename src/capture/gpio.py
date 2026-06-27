from capture.models import GPIOConfig


def create_trigger(config: GPIOConfig):
    from gpiozero import Device, DigitalInputDevice
    from gpiozero.pins.lgpio import LGPIOFactory

    Device.pin_factory = LGPIOFactory()
    return DigitalInputDevice(
        config.pin,
        pull_up=config.pull_up,
        bounce_time=config.bounce_time_seconds,
    )
