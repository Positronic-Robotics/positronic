import configuronic as cfn

from positronic.drivers.webxr import WebXR

oculus = cfn.Config(WebXR, port=5005, frontend='oculus')

# iPhone controller: open http://<server-ip>:5005/ on the phone in XR Browser
iphone = oculus.override(frontend='iphone', use_https=False, sensitivity=1.0)

# Android phones require https
android = oculus.override(frontend='iphone', use_https=True, sensitivity=2.0)
