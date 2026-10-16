import configuronic as cfn

from positronic.drivers.webxr import Frontend, WebXR


@cfn.config(port=5005, use_https=True, sensitivity=1.0, jpeg_quality=50)
def webxr(port: int, frontend: Frontend, use_https: bool, sensitivity: float, jpeg_quality: int):
    return WebXR(port=port, frontend=frontend, use_https=use_https, sensitivity=sensitivity, jpeg_quality=jpeg_quality)


oculus = webxr.override(frontend=Frontend.OCULUS)

# iPhone controller: open http://<server-ip>:5005/ on the phone in XR Browser
iphone = webxr.override(frontend=Frontend.IPHONE, use_https=False, sensitivity=2.0)

# Android phones require https
android = webxr.override(frontend=Frontend.IPHONE, use_https=True, sensitivity=2.0)

# The phone pages for `positronic teleop`: the same page, with momentary buttons that carry the teleop labels
iphone_teleop = iphone.override(frontend=Frontend.IPHONE_TELEOP)
android_teleop = android.override(frontend=Frontend.IPHONE_TELEOP)
