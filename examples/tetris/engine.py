"""Tetris engine: gravity, active piece, legal locks, host verify.

The host (this module) owns physics. A Choice scorer only picks among
placement IDs the engine already enumerated. Coordinates are zero-based:
columns 0..WIDTH-1 left to right; rows 0..HEIGHT-1 top to bottom.

Gravity interval shrinks with level (derived from lines cleared).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

WIDTH = 10
HEIGHT = 20
SPAWN_COL = 3
SPAWN_ROW = 0

# Base gravity: milliseconds between automatic soft-drops at level 0.
BASE_DROP_MS = 800
MIN_DROP_MS = 55
DROP_MS_PER_LEVEL = 70
LINES_PER_LEVEL = 10

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
LINE_SCORES = (0, 40, 100, 300, 1200)


@dataclass(frozen=True)
class Placement:
    """One legal final lock for the active piece."""

    placement_id: str
    piece: str
    rotation: str
    rotation_index: int
    column: int
    landing_row: int
    origin_col: int
    origin_row: int
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


@dataclass(frozen=True)
class ActivePiece:
    """Falling piece with an origin and rotation."""

    piece: str
    rotation_index: int
    origin_col: int
    origin_row: int

    @property
    def rotation(self) -> str:
        return ROTATION_LABELS[self.rotation_index % 4]

    def cells(self) -> tuple[tuple[int, int], ...]:
        return cells_for(self.piece, self.rotation_index, self.origin_col, self.origin_row)


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


def level_for_lines(lines_total: int) -> int:
    return max(0, lines_total // LINES_PER_LEVEL)


def drop_interval_ms(level: int, score: int = 0) -> int:
    """Gravity tempo: faster as level and score rise."""
    score_notches = max(0, score) // 50
    return max(
        MIN_DROP_MS,
        BASE_DROP_MS - level * DROP_MS_PER_LEVEL - score_notches * 30,
    )


def score_for_clear(lines_cleared: int, level: int) -> int:
    base = LINE_SCORES[lines_cleared] if lines_cleared <= 4 else LINE_SCORES[4]
    return base * (level + 1)


def score_for_soft_drop(rows: int) -> int:
    """Classic soft-drop points: one point per row fallen."""
    return max(0, rows)


def spawn_piece(board: Sequence[Sequence[str | None]], piece: str) -> ActivePiece | None:
    """Spawn at the standard column; return None if blocked (game over)."""
    if piece not in SHAPES:
        raise ValueError(f"unknown piece {piece!r}")
    active = ActivePiece(piece, 0, SPAWN_COL, SPAWN_ROW)
    if collides(board, active.cells()):
        return None
    return active


def try_shift(
    board: Sequence[Sequence[str | None]],
    active: ActivePiece,
    dx: int,
) -> ActivePiece | None:
    nxt = ActivePiece(active.piece, active.rotation_index, active.origin_col + dx, active.origin_row)
    if collides(board, nxt.cells()):
        return None
    return nxt


def try_rotate(
    board: Sequence[Sequence[str | None]],
    active: ActivePiece,
    *,
    clockwise: bool = True,
) -> ActivePiece | None:
    delta = 1 if clockwise else -1
    nxt_rot = (active.rotation_index + delta) % 4
    # Simple wall kicks: try origin, then ±1 / ±2 columns.
    for kick in (0, -1, 1, -2, 2):
        nxt = ActivePiece(active.piece, nxt_rot, active.origin_col + kick, active.origin_row)
        if not collides(board, nxt.cells()):
            return nxt
    return None


def soft_drop(
    board: Sequence[Sequence[str | None]],
    active: ActivePiece,
) -> tuple[ActivePiece | None, bool]:
    """Move one row down. Returns (piece_or_None_if_locked, locked)."""
    nxt = ActivePiece(active.piece, active.rotation_index, active.origin_col, active.origin_row + 1)
    if collides(board, nxt.cells()):
        return active, True
    return nxt, False


def hard_drop_row(
    board: Sequence[Sequence[str | None]],
    active: ActivePiece,
) -> ActivePiece:
    current = active
    while True:
        nxt, locked = soft_drop(board, current)
        if locked or nxt is None:
            return current
        current = nxt


def ghost_cells(
    board: Sequence[Sequence[str | None]],
    active: ActivePiece,
) -> tuple[tuple[int, int], ...]:
    return hard_drop_row(board, active).cells()


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
    centre = 4.5
    centre_penalty = abs((column + 0.5) - centre)
    depth_bonus = landing_row
    return (
        100.0 * lines_cleared
        - 40.0 * holes
        - 2.5 * aggregate_height
        - 1.5 * bumpiness
        - 0.35 * centre_penalty
        + 0.15 * depth_bonus
    )


def format_placement_id(
    piece: str,
    rotation: str,
    column: int,
    landing_row: int,
    *,
    disambiguator: int = 0,
) -> str:
    """Human-readable lock id: piece, rotation, column (and row if needed)."""
    base = f"{piece} rot={rotation} col={column}"
    if disambiguator <= 0:
        return base
    return f"{base} row={landing_row}"


def enumerate_placements(
    board: Sequence[Sequence[str | None]],
    piece: str,
) -> list[Placement]:
    """All legal final placements for ``piece`` on ``board``."""
    if piece not in SHAPES:
        raise ValueError(f"unknown piece {piece!r}")
    found: list[Placement] = []
    seen_cells: set[tuple[tuple[int, int], ...]] = set()
    used_ids: set[str] = set()
    for rotation_index in range(4):
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
            rotation = ROTATION_LABELS[rotation_index]
            placement_id = format_placement_id(piece, rotation, column, landing_row)
            if placement_id in used_ids:
                placement_id = format_placement_id(
                    piece, rotation, column, landing_row, disambiguator=1
                )
            suffix = 2
            while placement_id in used_ids:
                placement_id = (
                    f"{format_placement_id(piece, rotation, column, landing_row, disambiguator=1)}"
                    f"#{suffix}"
                )
                suffix += 1
            used_ids.add(placement_id)
            found.append(
                Placement(
                    placement_id=placement_id,
                    piece=piece,
                    rotation=rotation,
                    rotation_index=rotation_index,
                    column=column,
                    landing_row=landing_row,
                    origin_col=origin_col,
                    origin_row=origin_row,
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
            and item.origin_col == chosen.origin_col
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


def lock_active(
    board: Sequence[Sequence[str | None]],
    active: ActivePiece,
) -> tuple[list[list[str | None]], int]:
    return apply_lock(board, active.piece, active.cells())


def board_to_rows(board: Sequence[Sequence[str | None]]) -> list[str]:
    """Compact ASCII rows for traces ('.' empty, letter filled)."""
    return ["".join(cell if cell else "." for cell in row) for row in board]


def plan_approach(
    board: Sequence[Sequence[str | None]],
    piece: str,
    target: Placement,
) -> list[ActivePiece]:
    """Spawn → rotate → shift → fall, yielding each visible pose.

    Keeps gravity visible: every soft-drop is a step. Horizontal/rotation
    moves happen near the top so the fall reads clearly.
    """
    active = spawn_piece(board, piece)
    if active is None:
        return []
    path: list[ActivePiece] = [active]

    # Rotate toward the target (shortest direction).
    cur_rot = active.rotation_index
    target_rot = target.rotation_index % 4
    cw = (target_rot - cur_rot) % 4
    ccw = (cur_rot - target_rot) % 4
    clockwise = cw <= ccw
    turns = min(cw, ccw)
    for _ in range(turns):
        nxt = try_rotate(board, active, clockwise=clockwise)
        if nxt is None:
            break
        active = nxt
        path.append(active)

    # Shift toward target origin column while still high.
    while active.origin_col != target.origin_col:
        dx = 1 if target.origin_col > active.origin_col else -1
        nxt = try_shift(board, active, dx)
        if nxt is None:
            break
        active = nxt
        path.append(active)

    # Visible gravity until the piece rests on the board / stack.
    while True:
        nxt, locked = soft_drop(board, active)
        if locked:
            break
        assert nxt is not None
        active = nxt
        path.append(active)

    return path


class SevenBag:
    """Deterministic 7-bag with an explicit seed sequence for demos."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._bag: list[str] = []
        self._drawn = 0

    def _ensure_bag(self) -> None:
        if self._bag:
            return
        order = list(BAG_ORDER)
        state = (self._seed * 1103515245 + self._drawn * 12345) & 0x7FFFFFFF
        for i in range(len(order) - 1, 0, -1):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            j = state % (i + 1)
            order[i], order[j] = order[j], order[i]
        self._bag = order

    def peek(self) -> str:
        """Upcoming piece without consuming it (next-box preview)."""
        self._ensure_bag()
        return self._bag[-1]

    def next_piece(self) -> str:
        self._ensure_bag()
        self._drawn += 1
        return self._bag.pop()
