"""Check that a public image holds only public files: list its files, and search them for internal names.

Usage
    python3 utilities/check_public_image.py positro/molmospaces:local

It exports the filesystem of the image, and reports a finding for each of these:
- a file under /positronic that git does not track, other than build output;
- a path outside the public sources whose name has an internal word in it;
- a small file outside the public sources whose text has a key, an internal word or a private host name in it.

It exits 1 when it reports a finding. The public sources are the system directories, the Python environments,
the uv Pythons and the MolmoSpaces checkout.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = 'positronic/'

PUBLIC_PREFIXES = (
    'bin/',
    'boot/',
    'dev/',
    'lib/',
    'lib32/',
    'lib64/',
    'proc/',
    'sbin/',
    'sys/',
    'usr/',
    'var/lib/apt/',
    'var/lib/dpkg/',
    'var/cache/',
    'var/log/',
    'root/.local/share/uv/',
    # The MolmoSpaces checkout at its public pin (positronic/simulator/molmo_spaces/launcher.py).
    'root/.cache/positronic/molmospaces/src/',
)
PUBLIC_PARTS = ('.venv', 'site-packages', '.git')

# Generic words for a service that is not public. A public image names none of them outside its public sources.
INTERNAL_WORDS = re.compile(r'orchestrator|gateway|platform', re.IGNORECASE)
KEY_FILE_NAMES = re.compile(r'(^|/)(\.env|id_rsa|id_ed25519|id_ecdsa|credentials(\.json)?|[^/]*\.pem)$')
SECRET_TEXT = re.compile(
    r'-----BEGIN [A-Z ]*PRIVATE KEY-----'  # a private key
    r'|\bAKIA[0-9A-Z]{16}\b'  # an AWS access key id
    r'|\b[a-z0-9-]+\.ts\.net\b'  # a tailnet host name
    r'|\b[a-z0-9-]+\.(internal|corp|lan)\b'  # a private DNS name
)
MAX_TEXT_BYTES = 1 << 20


def is_public_source(path: str) -> bool:
    return path.startswith(PUBLIC_PREFIXES) or any(part in PUBLIC_PARTS for part in path.split('/'))


def is_build_output(path: str) -> bool:
    """A file that the install or an import of the repository writes: package metadata and byte code."""
    return any(part == '__pycache__' or part.endswith('.egg-info') for part in path.split('/'))


def tracked_files(repo: Path = REPO_ROOT) -> set[str]:
    listed = subprocess.run(['git', 'ls-files', '-z'], cwd=repo, capture_output=True, check=True).stdout
    return {APP_ROOT + name for name in listed.decode().split('\0') if name}


def findings(image: tarfile.TarFile, tracked: set[str]) -> Iterable[str]:
    """The findings in the filesystem of `image`, a tar read in order."""
    for member in image:
        path = member.name.removeprefix('./')
        if is_public_source(path):
            continue
        if path.startswith(APP_ROOT):
            if member.isfile() and path not in tracked and not is_build_output(path):
                yield f'/{path}: git does not track this file'
            continue
        if INTERNAL_WORDS.search(path) or KEY_FILE_NAMES.search(path):
            yield f'/{path}: an internal name'
        if member.isfile() and member.size <= MAX_TEXT_BYTES:
            file = image.extractfile(member)
            text = file.read().decode(errors='replace') if file else ''
            match = SECRET_TEXT.search(text) or INTERNAL_WORDS.search(text)
            if match:
                yield f'/{path}: holds {match.group(0)!r}'


def _exported(image: str) -> tuple[subprocess.Popen[bytes], str]:
    container = subprocess.run(
        ['docker', 'create', '--entrypoint', '/bin/true', image], capture_output=True, text=True, check=True
    ).stdout.strip()
    return subprocess.Popen(['docker', 'export', container], stdout=subprocess.PIPE), container


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    export, container = _exported(args[0])
    try:
        assert export.stdout is not None  # stdout=PIPE
        with tarfile.open(fileobj=export.stdout, mode='r|') as image:
            found = list(findings(image, tracked_files()))
        export.wait()
    finally:
        subprocess.run(['docker', 'rm', container], capture_output=True, check=False)
    for finding in found:
        print(finding)
    print(f'{args[0]}: {len(found)} findings')
    return 1 if found else 0


if __name__ == '__main__':
    raise SystemExit(main())
