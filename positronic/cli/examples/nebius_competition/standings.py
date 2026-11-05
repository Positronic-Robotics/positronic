"""Print the competition leaderboards, or the standings on one of them.

    uv run positronic/cli/examples/nebius_competition/standings.py
    uv run positronic/cli/examples/nebius_competition/standings.py --board=<slug>

No key is needed: a published leaderboard is readable by anyone. With no `--board`, the script lists the
leaderboards that `leaderboards.list` returns for the tag of the competition site, with the eval each one
ranks. With a slug, it prints every row of that leaderboard: place, display name, tag, the score that the
leaderboard ranks by, and submission id.

The platform is `--platform-url`, else POSITRONIC_PLATFORM_URL, else the one that
`positronic account register` saved, as in `submit_sample.py`.

The platform owns the set of leaderboards and evals. Read the names here; do not copy them from a document.
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from platform_client.client import PlatformClient
from platform_client.config import platform_url_from, record_if_needed
from platform_client.enums import ErrorCode
from platform_client.errors import PlatformError
from platform_client.leaderboards import LeaderboardDescriptor, LeaderboardRef, Metric

from positronic.cli.examples.nebius_competition.leaderboards import LEADERBOARD_TAG


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """The header and the rows as aligned columns; the last column is not padded."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    return [
        '  '.join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip()
        for line in [header, *rows]
    ]


def _ranked_metric(descriptor: LeaderboardDescriptor) -> Metric:
    return next(metric for metric in descriptor.metrics if metric.key == descriptor.ranked_by)


def run(client: PlatformClient, board: LeaderboardRef | None) -> list[str]:
    """The lines to print: the leaderboards on offer, or the standings on `board`."""
    if board is None:
        listed = client.list_leaderboards(LEADERBOARD_TAG).leaderboards
        if not listed:
            return ['no leaderboards']
        rows = [
            [
                summary.leaderboard,
                summary.descriptor.eval or '-',
                _ranked_metric(summary.descriptor).name,
                summary.title,
            ]
            for summary in listed
        ]
        return _table(['board', 'eval', 'ranked by', 'title'], rows)
    try:
        response = client.get_leaderboard(board)
    except PlatformError as exc:
        if exc.code is not ErrorCode.not_found:
            raise
        listed = client.list_leaderboards(LEADERBOARD_TAG).leaderboards
        offered = ', '.join(summary.leaderboard for summary in listed)
        raise SystemExit(f'{exc.message}: {board}\nleaderboards on offer: {offered}') from exc
    metric = _ranked_metric(response.descriptor)
    ranks = f'ranks {response.descriptor.eval} by' if response.descriptor.eval is not None else 'ranks by'
    header = f'{response.leaderboard}: {ranks} {metric.name}'
    if not response.rows:
        return [header, 'no entries']
    rows = []
    for row in response.rows:
        score = row.metrics.get(metric.key)
        rows.append([
            'baseline' if row.rank is None else str(row.rank),
            f'{row.display_name}#{row.user_tag}' if row.user_tag is not None else row.display_name,
            '-' if score is None else f'{score.value:.{metric.decimals}f}',
            '-' if row.submission is None else str(row.submission.id),
        ])
    return [header, *_table(['rank', 'name#tag', metric.name, 'submission'], rows)]


def anonymous_client(platform_url: str | None = None, *, client: httpx.Client | None = None) -> PlatformClient:
    """A client that sends no key, whatever the environment holds: a published leaderboard is readable by anyone."""
    anonymous = PlatformClient(platform_url, client=client)
    anonymous.api_key = None
    return anonymous


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform-url', default=None, help='a platform other than the default one')
    parser.add_argument('--board', default=None, help='the slug of one leaderboard; the listing prints the slugs')
    args = parser.parse_args(argv)
    # Every value the wire types refuse — a slug, a platform URL — is refused here, before any request.
    try:
        board = LeaderboardRef(args.board) if args.board is not None else None
        record = record_if_needed(os.environ, args.platform_url, key_required=False)
        client = anonymous_client(platform_url_from(os.environ, args.platform_url, record))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with client:
        try:
            lines = run(client, board)
        except PlatformError as exc:
            raise SystemExit(f'{exc.code.name}: {exc.message}') from exc
    sys.stdout.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
