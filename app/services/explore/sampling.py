"""Projection point sampling and density grids."""

from __future__ import annotations

import zlib

import numpy as np

SAMPLE_GRID = 256


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def priority_permutation(n: int, seed: int) -> np.ndarray:
    """Deterministic random rank per row (lower = kept first)."""
    rng = np.random.default_rng(seed)
    ranks = np.empty(n, dtype=np.int32)
    ranks[rng.permutation(n)] = np.arange(n, dtype=np.int32)
    return ranks


def bounds_of(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float] | None:
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return None
    xs, ys = x[finite], y[finite]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def grid_cells(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> np.ndarray:
    """Cell index (row-major, x then y) for each point; caller ensures finite."""
    xmin, xmax, ymin, ymax = bounds
    sx = (xmax - xmin) or 1.0
    sy = (ymax - ymin) or 1.0
    ix = np.clip(((x - xmin) / sx * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - ymin) / sy * ny).astype(np.int64), 0, ny - 1)
    return iy * nx + ix


def stratified_sample(
    x: np.ndarray,
    y: np.ndarray,
    cap: int,
    priority: np.ndarray,
    grid: int = SAMPLE_GRID,
) -> np.ndarray:
    """Indices of at most ``cap`` finite points, spread evenly over a grid.

    Picks the largest per-cell quota ``c`` whose total fits the budget, then
    spends any leftover budget on the next-ranked point of the busiest cells, so
    sparse clusters and outliers survive while dense regions are thinned.
    Returns every finite point when there are no more than ``cap``.
    """
    finite_idx = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if finite_idx.size <= cap:
        return finite_idx.astype(np.int64)
    if cap <= 0:
        return np.empty(0, dtype=np.int64)

    fx = x[finite_idx].astype(np.float64)
    fy = y[finite_idx].astype(np.float64)
    bounds = bounds_of(fx, fy)
    assert bounds is not None
    cells = grid_cells(fx, fy, bounds, grid, grid)
    prio = priority[finite_idx]

    order = np.lexsort((prio, cells))
    cells_sorted = cells[order]
    counts = np.bincount(cells, minlength=grid * grid)
    starts = np.cumsum(counts) - counts
    rank = np.arange(order.size, dtype=np.int64) - starts[cells_sorted]

    lo, hi = 0, int(counts.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(np.minimum(counts, mid).sum()) <= cap:
            lo = mid
        else:
            hi = mid - 1
    quota = lo
    chosen = order[rank < quota]
    leftover = cap - chosen.size
    if leftover > 0:
        extra = order[rank == quota]
        if extra.size > leftover:
            extra = extra[np.argsort(prio[extra], kind="stable")[:leftover]]
        chosen = np.concatenate([chosen, extra])
    return np.sort(finite_idx[chosen]).astype(np.int64)


def density_grid(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> np.ndarray:
    """uint32 counts, shape [ny, nx] (row = y bin)."""
    if x.size == 0:
        return np.zeros((ny, nx), dtype=np.uint32)
    cells = grid_cells(x.astype(np.float64), y.astype(np.float64), bounds, nx, ny)
    return np.bincount(cells, minlength=nx * ny).astype(np.uint32).reshape(ny, nx)
