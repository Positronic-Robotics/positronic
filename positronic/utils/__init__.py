import socket
from pathlib import Path

from positronic import __file__ as pkg_init_file


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


def merge_dicts(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            dst[key] = merge_dicts(dst[key], value)
        else:
            dst[key] = value
    return dst


def flatten_dict(d: dict, prefix: str = '') -> dict:
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result.update(flatten_dict(value, f'{prefix}{key}.'))
        else:
            result[f'{prefix}{key}'] = value
    return result


def package_assets_path(relative_path: str) -> str:
    pdg_dir = Path(pkg_init_file).resolve().parent
    return str(pdg_dir / relative_path)
