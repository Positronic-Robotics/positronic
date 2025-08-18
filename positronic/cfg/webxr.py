import configuronic as cfn
from positronic.drivers.webxr import WebXR

oculus = cfn.Config(
    WebXR,
    port=5005,
    ssl_keyfile="key.pem",
    ssl_certfile="cert.pem",
    default_frontend="oculus",  # or "iphone"
)

iphone = oculus.override(default_frontend="iphone")
