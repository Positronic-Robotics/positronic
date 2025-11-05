import configuronic as cfn


@cfn.config(port=5005, use_https=True, sensitivity=1.0)
def webxr(port: int, frontend: str, use_https: bool, sensitivity: float):
    from positronic.drivers.webxr import WebXR

    return WebXR(port=port, frontend=frontend, use_https=use_https, sensitivity=sensitivity)


oculus = webxr.override(frontend='oculus')

# iPhone controller: open http://<server-ip>:5005/ on the phone in XR Browser
iphone = webxr.override(frontend='iphone', use_https=False, sensitivity=1.0)

# Android phones require https
android = webxr.override(frontend='iphone', use_https=True, sensitivity=2.0)
