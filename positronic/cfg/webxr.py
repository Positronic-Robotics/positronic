import configuronic as cfn

from positronic.drivers.webxr import Frontend, WebXR

webxr = cfn.Config(WebXR, port=5005)


oculus = webxr.override(frontend=Frontend.OCULUS)

# iPhone controller: open http://<server-ip>:5005/ on the phone in XR Browser
iphone = webxr.override(frontend=Frontend.IPHONE, use_https=False, sensitivity=2.0)

# Android phones require https
android = webxr.override(frontend=Frontend.IPHONE, use_https=True, sensitivity=2.0)

# The phone pages for `positronic teleop`: the same page, with momentary buttons that carry the teleop labels
iphone_teleop = iphone.override(frontend=Frontend.IPHONE_TELEOP)
android_teleop = android.override(frontend=Frontend.IPHONE_TELEOP)
