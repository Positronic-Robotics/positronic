"""Print the public leaderboards, or the standings on one of them.

    uv run positronic/cli/examples/nebius_competition/standings.py
    uv run positronic/cli/examples/nebius_competition/standings.py --board=<slug>

No key is needed: a public board is readable by anyone. With no `--board`, the script lists the
boards `rankings.list` returns, with the eval each one ranks. With a slug, it prints every row of
that board: rank, display name, tag, primary score and submission id.

The platform owns the set of boards and evals. Read the names here; do not copy them from a document.
"""

from __future__ import annotations

import argparse
import sys

from platform_client.boards import BoardRef
from platform_client.client import PlatformClient
from platform_client.enums import ErrorCode
from platform_client.errors import PlatformError
from platform_client.responses import BoardSummary, RankingRow, RankingsResponse


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """The header and the rows as aligned columns; the last column is not padded."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    return [
        '  '.join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip()
        for line in [header, *rows]
    ]


def board_lines(boards: list[BoardSummary]) -> list[str]:
    """One line per board: its slug, the eval it ranks, the metric it sorts on, and its title."""
    if not boards:
        return ['no boards']
    rows = [[board.board, board.eval, board.primary_metric, board.title] for board in boards]
    return _table(['board', 'eval', 'primary metric', 'title'], rows)


def _row(row: RankingRow) -> list[str]:
    score = '-' if row.scores.primary is None else f'{row.scores.primary:.3f}'
    return [str(row.rank), f'{row.display_name}#{row.tag}', score, str(row.submission_id)]


def standings_lines(response: RankingsResponse) -> list[str]:
    """A header naming the board, then one line per row."""
    lines = [f'{response.board}: ranks {response.eval} by {response.primary_metric}']
    if not response.rankings:
        return [*lines, 'no entries']
    return [
        *lines,
        *_table(['rank', 'name#tag', response.primary_metric, 'submission'], list(map(_row, response.rankings))),
    ]


def run(client: PlatformClient, board: BoardRef | None) -> list[str]:
    """The lines to print: the boards on offer, or the standings on `board`."""
    if board is None:
        return board_lines(client.list_boards().boards)
    try:
        return standings_lines(client.rankings(board=board))
    except PlatformError as exc:
        if exc.code is not ErrorCode.not_found:
            raise
        offered = ', '.join(summary.board for summary in client.list_boards().boards)
        raise SystemExit(f'{exc.message}: {board}\nboards on offer: {offered}') from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform-url', default=None, help='a platform other than the default one')
    parser.add_argument('--board', default=None, help='the slug of one board; the listing prints the slugs')
    args = parser.parse_args(argv)
    try:
        board = BoardRef(args.board) if args.board is not None else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with PlatformClient(args.platform_url) as client:
        try:
            lines = run(client, board)
        except PlatformError as exc:
            raise SystemExit(f'{exc.code.name}: {exc.message}') from exc
    sys.stdout.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
