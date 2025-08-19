import configuronic as cfn
from positronic.drivers.webxr import WebXR

oculus = cfn.Config(WebXR, port=5005, ssl_keyfile="key.pem", ssl_certfile="cert.pem", frontend="oculus")

iphone = oculus.override(frontend="iphone", use_https=False)
