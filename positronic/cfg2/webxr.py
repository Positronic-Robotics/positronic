import ironic as ir
from pimm.drivers.webxr import WebXR

webxr = ir.Config(WebXR, port=8000)
