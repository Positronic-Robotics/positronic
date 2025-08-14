import configuronic as cfn
from positronic.drivers.webxr import WebXR

webxr = cfn.Config(WebXR, port=5005, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
