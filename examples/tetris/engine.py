"""Minimal Tetris engine: enumerate legal placements, verify, then lock.

The host (this module) owns physics. A Choice scorer only picks among
placement IDs the engine already enumerated. Coordinates are zero-based:
columns 0..WIDTH-1 left to right; rows 0..HEIGHT-1 top to bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

WIDTH = 10
HEIGHT = 20

# SRS-style shapes: each rotation is a tuple of (col, row) offsets from a
# reference cell. Rotations cycle 0 → R → 2 → L.
SHAPES: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "I": (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((1, 0), (1, 1), (1, 2), (1, 3)),
    ),
    "O": (
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
    ),
    "T": (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "S": (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((1, 1), (2, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "Z": (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
    "J": (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    "L": (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
}

ROTATION_LABELS = ("0", "R", "2", "L")
BAG_ORDER = ("I", "O", "T", "S", "Z", "J", "L")


@dataclass(frozen=True)
class Placement:
    """One legal final lock for the active piece."""

    placement_id: str
    piece: str
    rotation: str
    rotation_index: int
    column: int
    landing_row: int
    occupied_cells: tuple[tuple[int, int], ...]
    lines_cleared: int
    holes: int
    aggregate_height: int
    bumpiness: int
    heuristic_score: float


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str
    placement: Placement | None = None


def empty_board() -> list[list[str | None]]:
    return [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]


def clone_board(board: Sequence[Sequence[str | None]]) -> list[list[str | None]]:
    return [list(row) for row in board]


def cells_for(piece: str, rotation_index: int, origin_col: int, origin_row: int) -> tuple[tuple[int, int], ...]:
    shape = SHAPES[piece][rotation_index % 4]
    return tuple((origin_col + dc, origin_row + dr) for dc, dr in shape)


def in_bounds(col: int, row: int) -> bool:
    return 0 <= col < WIDTH and 0 <= row < HEIGHT


def collides(board: Sequence[Sequence[str | None]], cells: Iterable[tuple[int, int]]) -> bool:
    for col, row in cells:
        if not in_bounds(col, row):
            return True
        if board[row][col] is not None:
            return True
    return False


def drop_origin(
    board: Sequence[Sequence[str | None]],
    piece: str,
    rotation_index: int,
    origin_col: int,
) -> int | None:
    """Return the lowest origin_row where the piece fits, or None."""
    origin_row = 0
    cells = cells_for(piece, rotation_index, origin_col, origin_row)
    if collides(board, cells):
        # Try spawning slightly higher is impossible; reject this column.
        return None
    while True:
        next_cells = cells_for(piece, rotation_index, origin_col, origin_row + 1)
        if collides(board, next_cells):
            return origin_row
        origin_row += 1


def _column_heights(board: Sequence[Sequence[str | None]]) -> list[int]:
    heights: list[int] = []
    for col in range(WIDTH):
        height = 0
        for row in range(HEIGHT):
            if board[row][col] is not None:
                height = HEIGHT - row
                break
        heights.append(height)
    return heights


def _count_holes(board: Sequence[Sequence[str | None]]) -> int:
    holes = 0
    for col in range(WIDTH):
        seen_block = False
        for row in range(HEIGHT):
            cell = board[row][col]
            if cell is not None:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def _bumpiness(heights: Sequence[int]) -> int:
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def apply_lock(
    board: Sequence[Sequence[str | None]],
    piece: str,
    cells: Sequence[tuple[int, int]],
) -> tuple[list[list[str | None]], int]:
    next_board = clone_board(board)
    for col, row in cells:
        next_board[row][col] = piece
    # Keep non-full rows (rows that still have an empty cell).
    kept = [row for row in next_board if any(cell is None for cell in row)]
    cleared = HEIGHT - len(kept)
    pad = [[None for _ in range(WIDTH)] for _ in range(cleared)]
    return pad + kept, cleared


def board_metrics(
    board: Sequence[Sequence[str | None]],
) -> tuple[int, int, int]:
    heights = _column_heights(board)
    return _count_holes(board), sum(heights), _bumpiness(heights)


def score_placement(
    lines_cleared: int,
    holes: int,
    aggregate_height: int,
    bumpiness: int,
    *,
    column: int = 0,
    landing_row: int = 0,
) -> float:
    """Deterministic Path A heuristic (higher is better)."""
    # Prefer clearing lines and packing low; mild centre bias for empty boards.
    centre = 4.5
    centre_penalty = abs((column + 0.5) - centre)
    depth_bonus = landing_row  # larger row index = lower on the board
    return (
        100.0 * lines_cleared
        - 40.0 * holes
        - 2.5 * aggregate_height
        - 1.5 * bumpiness
        - 0.35 * centre_penalty
        + 0.15 * depth_bonus
    )


def enumerate_placements(
    board: Sequence[Sequence[str | None]],
    piece: str,
) -> list[Placement]:
    """All legal final placements for ``piece`` on ``board``."""
    if piece not in SHAPES:
        raise ValueError(f"unknown piece {piece!r}")
    found: list[Placement] = []
    seen_cells: set[tuple[tuple[int, int], ...]] = set()
    for rotation_index in range(4):
        # Origin columns that can possibly fit the shape width.
        for origin_col in range(-2, WIDTH):
            origin_row = drop_origin(board, piece, rotation_index, origin_col)
            if origin_row is None:
                continue
            cells = cells_for(piece, rotation_index, origin_col, origin_row)
            if collides(board, cells):
                continue
            key = tuple(sorted(cells))
            if key in seen_cells:
                continue
            seen_cells.add(key)
            next_board, lines_cleared = apply_lock(board, piece, cells)
            holes, aggregate_height, bumpiness = board_metrics(next_board)
            column = min(col for col, _ in cells)
            landing_row = min(row for _, row in cells)
            placement_id = f"p{len(found)}"
            found.append(
                Placement(
                    placement_id=placement_id,
                    piece=piece,
                    rotation=ROTATION_LABELS[rotation_index],
                    rotation_index=rotation_index,
                    column=column,
                    landing_row=landing_row,
                    occupied_cells=cells,
                    lines_cleared=lines_cleared,
                    holes=holes,
                    aggregate_height=aggregate_height,
                    bumpiness=bumpiness,
                    heuristic_score=score_placement(
                        lines_cleared,
                        holes,
                        aggregate_height,
                        bumpiness,
                        column=column,
                        landing_row=landing_row,
                    ),
                )
            )
    return found


def verify_placement(
    board: Sequence[Sequence[str | None]],
    piece: str,
    placement_id: str,
    candidates: Sequence[Placement],
) -> VerifyResult:
    """Host check: ID must still be legal with identical cells."""
    by_id = {item.placement_id: item for item in candidates}
    chosen = by_id.get(placement_id)
    if chosen is None:
        return VerifyResult(False, f"unknown placement id {placement_id!r}")
    if chosen.piece != piece:
        return VerifyResult(False, "placement piece does not match active piece")
    fresh = enumerate_placements(board, piece)
    match = next(
        (
            item
            for item in fresh
            if item.occupied_cells == chosen.occupied_cells
            and item.rotation_index == chosen.rotation_index
        ),
        None,
    )
    if match is None:
        return VerifyResult(False, "placement cells are no longer legal")
    return VerifyResult(True, "host verified rotation, column, and cells", match)


def lock_verified(
    board: Sequence[Sequence[str | None]],
    piece: str,
    verified: Placement,
) -> tuple[list[list[str | None]], int]:
    return apply_lock(board, piece, verified.occupied_cells)


def board_to_rows(board: Sequence[Sequence[str | None]]) -> list[str]:
    """Compact ASCII rows for traces ('.' empty, letter filled)."""
    return ["".join(cell if cell else "." for cell in row) for row in board]


class SevenBag:
    """Deterministic 7-bag with an explicit seed sequence for demos."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._bag: list[str] = []
        self._drawn = 0

    def next_piece(self) -> str:
        if not self._bag:
            # Simple deterministic shuffle from seed + draw count.
            order = list(BAG_ORDER)
            state = (self._seed * 1103515245 + self._drawn * 12345) & 0x7FFFFFFF
            for i in range(len(order) - 1, 0, -1):
                state = (1103515245 * state + 12345) & 0x7FFFFFFF
                j = state % (i + 1)
                order[i], order[j] = order[j], order[i]
            self._bag = order
        self._drawn += 1
        return self._bag.pop()
