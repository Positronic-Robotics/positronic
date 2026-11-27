import configuronic as cfn

from positronic.drivers.webxr import Frontend, SessionMode, WebXR

webxr = cfn.Config(WebXR, port=5005)


oculus = webxr.override(frontend=Frontend.HEADSET)
psvr2 = webxr.override(frontend=Frontend.HEADSET, session_mode=SessionMode.VR)

# iPhone controller: open http://<server-ip>:5005/ on the phone in XR Browser
iphone = webxr.override(frontend=Frontend.PHONE, use_https=False, sensitivity=2.0)

# Android phones require https
android = webxr.override(frontend=Frontend.PHONE, use_https=True, sensitivity=2.0)
