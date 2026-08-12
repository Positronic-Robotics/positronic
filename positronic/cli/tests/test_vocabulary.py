"""The platform's vocabulary is customer-neutral: one engagement's words stay in its own directory.

Both trees the platform's users read are held to it — the wire contract in `client/`, and the
commands and examples here — since a word that leaks into either is a word every other customer
reads about themselves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The words that name a role or a period in one engagement rather than anything the platform has.
# What the contract says instead is `user` for the account and `caller` for whoever is asking.
ENGAGEMENT_WORDS = (
    'competition',
    'participant',
    'contestant',
    'season',
    'qualifier',
    'prize',
    'entrant',
    'leaderboard',
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_ROOT = REPO_ROOT / 'positronic' / 'cli'
# Every surface a platform user reads: the wire contract, the commands that drive it, the examples.
# The rest of `positronic/cli` is robotics, whose own words are none of this rule's business — and
# the last test below is what keeps that line honest as commands move between the two.
SCANNED_ROOTS = (
    REPO_ROOT / 'client',
    CLI_ROOT / 'account',
    CLI_ROOT / 'examples',
    CLI_ROOT / 'eval' / 'run.py',
    CLI_ROOT / 'eval' / 'submit.py',
    CLI_ROOT / 'eval' / 'submissions.py',
)
ENGAGEMENT_DIR = REPO_ROOT / 'positronic' / 'cli' / 'examples' / 'nebius_competition'
TEXT_SUFFIXES = frozenset({'.py', '.md', '.toml'})

# No trailing boundary, so a plural is caught too. A leading one spares the engagement directory's
# own name where a path names it.
_ENGAGEMENT_WORD = re.compile(r'\b(' + '|'.join(ENGAGEMENT_WORDS) + ')', re.IGNORECASE)


def scanned_files() -> list[Path]:
    return sorted(
        path
        for root in SCANNED_ROOTS
        for path in ([root] if root.is_file() else root.rglob('*'))
        if path.suffix in TEXT_SUFFIXES and ENGAGEMENT_DIR not in path.parents and path != Path(__file__).resolve()
    )


@pytest.mark.parametrize('path', scanned_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_a_file_outside_the_engagement_directory_uses_none_of_its_words(path: Path):
    hits = [
        f'{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}'
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if _ENGAGEMENT_WORD.search(line)
    ]
    assert not hits, 'engagement vocabulary outside its directory:\n' + '\n'.join(hits)


def test_the_scan_reaches_both_trees_and_stops_at_the_engagement_directory():
    scanned = scanned_files()
    assert REPO_ROOT / 'client' / 'platform_client' / 'responses.py' in scanned
    assert CLI_ROOT / 'eval' / 'submissions.py' in scanned
    assert CLI_ROOT / 'examples' / 'walkthrough.py' in scanned
    assert ENGAGEMENT_DIR / 'submit_sample.py' not in scanned


def test_every_command_that_speaks_to_the_platform_is_scanned():
    # The roots above are a list, and a command moving between `eval` and `platform` would silently
    # leave it. What identifies a platform-facing module is that it imports the contract.
    speaks_to_the_platform = {
        path
        for path in CLI_ROOT.rglob('*.py')
        # Test support is exempt for the same reason the scan skips this file, and the engagement's
        # own directory is the one place its words belong.
        if 'platform_client' in path.read_text()
        and 'tests' not in path.parts
        and path.name != 'conftest.py'
        and ENGAGEMENT_DIR not in path.parents
    }
    assert speaks_to_the_platform <= set(scanned_files())
