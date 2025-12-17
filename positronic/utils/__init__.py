import importlib.metadata
import os
import platform
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from positronic import __file__ as pkg_init_file
from positronic.utils.checkpoints import get_latest_checkpoint
from positronic.utils.frozen_dict import frozen_keys_dict, frozen_view
from positronic.utils.git import get_git_diff, get_git_state


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


def find_uv_lock(start_dir: Path) -> Path | None:
    """Find uv.lock file in the given directory or its parent directories.

    Args:
        start_dir: Directory to start searching from.

    Returns:
        Path to uv.lock if found, None otherwise.
    """
    for parent in [start_dir] + list(start_dir.parents):
        potential_lock = parent / 'uv.lock'
        if potential_lock.exists():
            return potential_lock
    return None


def get_docker_info() -> dict | None:
    """Detect if running inside Docker and gather container metadata.

    Returns:
        Dictionary with Docker info if running in a container, None otherwise.
        May include: container_id, hostname, and cgroup_info.
    """
    docker_info = {}

    # Check for /.dockerenv file (common Docker indicator)
    if Path('/.dockerenv').exists():
        docker_info['dockerenv_present'] = True

    # Check IS_DOCKER environment variable (used in some setups)
    if os.environ.get('IS_DOCKER'):
        docker_info['is_docker_env'] = True

    # Try to read container ID from cgroup
    cgroup_path = Path('/proc/self/cgroup')
    if cgroup_path.exists():
        try:
            cgroup_content = cgroup_path.read_text(encoding='utf-8')
            # Look for docker or containerd in cgroup
            for line in cgroup_content.splitlines():
                if 'docker' in line or 'containerd' in line:
                    docker_info['in_container'] = True
                    # Try to extract container ID from cgroup path
                    # Format is typically: .../docker/<container_id> or .../docker-<container_id>.scope
                    parts = line.split('/')
                    for part in parts:
                        if part.startswith('docker-') and part.endswith('.scope'):
                            container_id = part.replace('docker-', '').replace('.scope', '')
                            docker_info['container_id'] = container_id[:12]  # Short format
                            break
                        elif len(part) == 64 and all(c in '0123456789abcdef' for c in part):
                            # Full container ID (64 hex chars)
                            docker_info['container_id'] = part[:12]  # Short format
                            break
                    break
        except (OSError, UnicodeDecodeError):
            pass

    # If we found any Docker indicators, add hostname
    if docker_info:
        try:
            docker_info['hostname'] = socket.gethostname()
        except OSError:
            pass
        return docker_info
    return None


def run_metadata(patterns: list[str] | None = None, add_git_diff: bool = True, add_uv_lock: bool = True) -> dict:
    """Capture script run metadata for maximum reproducibility.

    This function collects comprehensive information about the current script execution,
    including command-line arguments, git state, git diff, environment details, and
    uv virtual environment information.

    Args:
        patterns: List of file patterns for git diff (e.g., ['*.py', '*.toml']).
                 Defaults to ['*.py'] if None.
        add_git_diff: Whether to include git diff in the metadata.
        add_uv_lock: Whether to include uv.lock in the metadata.

    Returns:
        Dictionary containing:
        - timestamp_ns: Current time in nanoseconds
        - timestamp_iso: Current time in ISO format
        - command: Raw command-line arguments (sys.argv)
        - python: Python version
        - platform: Platform string
        - package_version: Positronic package version (if available)
        - git: Git state (commit, branch, dirty flag)
        - git_diff: Git diff for uncommitted changes matching patterns
        - environment: Environment information (VIRTUAL_ENV, uv.lock presence, docker info)
    """

    if patterns is None:
        patterns = ['*.py']

    metadata = {}
    metadata['timestamp_ns'] = time.time_ns()
    metadata['timestamp_iso'] = datetime.now(UTC).isoformat()
    metadata['command'] = sys.argv.copy()
    metadata['python'] = sys.version.split(' ')[0]
    metadata['platform'] = platform.platform()
    metadata['positronic_version'] = importlib.metadata.version('positronic')

    pkg_dir = Path(pkg_init_file).resolve().parent

    # Check if package directory is in a different git repo
    pkg_git_state = get_git_state(workdir=pkg_dir)
    if pkg_git_state:
        metadata['git.positronic'] = pkg_git_state
        if add_git_diff:
            git_diff = get_git_diff(workdir=pkg_dir, patterns=patterns)
            if git_diff:
                metadata['git.positronic.diff'] = git_diff

    git_state = get_git_state()
    if git_state and git_state != pkg_git_state:
        metadata['git.current'] = git_state
        if add_git_diff:
            git_diff = get_git_diff(patterns=patterns)
            if git_diff:
                metadata['git.current.diff'] = git_diff

    # Environment information
    environment = {}
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        environment['virtual_env'] = virtual_env

    # Docker information
    docker_info = get_docker_info()
    if docker_info:
        environment['docker'] = docker_info

    if add_uv_lock:
        # Read uv.lock content from current directory
        current_dir = Path.cwd()
        uv_lock_path = find_uv_lock(current_dir)
        uv_lock_content = uv_lock_path.read_text(encoding='utf-8') if uv_lock_path is not None else None
        if uv_lock_content:
            environment['uv_lock'] = uv_lock_content

        # Read uv.lock content from positronic package directory
        pkg_uv_lock_path = find_uv_lock(pkg_dir)
        if pkg_uv_lock_path != uv_lock_path and pkg_uv_lock_path is not None:
            pkg_uv_lock_content = pkg_uv_lock_path.read_text(encoding='utf-8')
            if pkg_uv_lock_content:
                environment['uv_lock.positronic'] = pkg_uv_lock_content

    if environment:
        metadata['environment'] = environment

    return metadata


def save_run_metadata(
    output_dir: str | Path,
    patterns: list[str] | None = None,
    add_git_diff: bool = True,
    add_uv_lock: bool = True,
    prefix: str = 'run_metadata',
) -> Path:
    """Capture and save script run metadata to a YAML file.

    Args:
        output_dir: Directory where the metadata YAML file should be saved.
        patterns: List of file patterns for git diff (e.g., ['*.py', '*.toml']).
                 Defaults to ['*.py'] if None.
        add_git_diff: Whether to include git diff in the metadata.
        add_uv_lock: Whether to include uv.lock in the metadata.
        prefix: Prefix for the metadata filename (default: 'run_metadata').
                The filename will be '{prefix}_{timestamp}.yaml'.

    Returns:
        Path to the saved metadata file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'{prefix}_{timestamp}.yaml'
    metadata = run_metadata(patterns, add_git_diff, add_uv_lock)

    # Custom dumper to format multiline strings (like git diffs) in readable YAML literal block style.
    # By default, PyYAML's emitter analyzes string content and may override the requested style
    # when it detects special characters (colons, quotes, etc.). We override choose_scalar_style()
    # to respect explicitly set styles, ensuring multiline strings use '|-' literal block format
    # instead of quoted strings with escaped newlines.
    class LiteralDumper(yaml.SafeDumper):
        def choose_scalar_style(self):
            # Respect explicitly set style (e.g., '|' for literal block)
            if self.event.style:
                return self.event.style
            return super().choose_scalar_style()

    def str_representer(dumper, data):
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    LiteralDumper.add_representer(str, str_representer)

    with output_path.open('w', encoding='utf-8') as f:
        yaml.dump(metadata, f, Dumper=LiteralDumper, default_flow_style=False, sort_keys=False)

    return output_path
