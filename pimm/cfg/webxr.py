import configuronic as cfn
from pimm.drivers.webxr import WebXR

webxr = cfn.Config(WebXR, port=5005, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
