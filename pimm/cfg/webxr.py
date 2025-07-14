import configuronic as cfgc
from pimm.drivers.webxr import WebXR

webxr = cfgc.Config(WebXR, port=5005, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
