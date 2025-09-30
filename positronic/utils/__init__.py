import socket


def resolve_host_ip() -> str:
    """Best-effort attempt at finding a non-loopback host IP for banner output."""
    candidates: list[str] = []

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connecting to a non-routable address reveals the outbound interface IP.
            s.connect(('10.255.255.255', 1))
            candidates.append(s.getsockname()[0])
    except OSError:
        pass

    for name in (socket.gethostname(), 'localhost'):
        try:
            candidates.append(socket.gethostbyname(name))
        except OSError:
            continue

    for ip in candidates:
        if ip and not ip.startswith('127.') and ip != '0.0.0.0':
            return ip

    return candidates[0] if candidates else '127.0.0.1'
