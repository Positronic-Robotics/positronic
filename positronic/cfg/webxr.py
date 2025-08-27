import configuronic as cfn


@cfn.config(
    port=5005,
    ssl_keyfile="key.pem",
    ssl_certfile="cert.pem",
    frontend="oculus",
    use_https=True,
)
def oculus(port: int, ssl_keyfile: str, ssl_certfile: str, use_https: bool, frontend: str):
    from positronic.drivers.webxr import WebXR
    return WebXR(port=port, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile, frontend=frontend, use_https=use_https)


# iPhone controller: open http://<server-ip>:5005/ on the phone in XR Browser
iphone = oculus.override(frontend="iphone", use_https=False)
