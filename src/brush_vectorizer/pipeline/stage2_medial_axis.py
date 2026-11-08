"""Stage 2: Euclidean Distance Transform & Medial Axis Extraction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from brush_vectorizer.config import PhysicalCalibrationConfig


@dataclass
class Stage2Result:
    """Outputs of Stage 2 Euclidean Distance Transform & Medial Axis Extraction."""

    edt_radius_px: np.ndarray     # R(x, y) Euclidean distance field in pixels
    raw_skeleton: np.ndarray      # 1-pixel binary skeleton K before spur pruning
    pruned_skeleton: np.ndarray   # 1-pixel binary skeleton K after spur pruning
    pruned_branch_count: int      # Number of spur branches pruned


def _build_zhang_suen_luts() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute 256-entry boolean lookup tables for Zhang-Suen and Holt staircase passes."""
    lut1 = np.zeros(256, dtype=bool)
    lut2 = np.zeros(256, dtype=bool)
    holt1 = np.zeros(256, dtype=bool)
    holt2 = np.zeros(256, dtype=bool)

    for idx in range(256):
        p2 = (idx >> 0) & 1
        p3 = (idx >> 1) & 1
        p4 = (idx >> 2) & 1
        p5 = (idx >> 3) & 1
        p6 = (idx >> 4) & 1
        p7 = (idx >> 5) & 1
        p8 = (idx >> 6) & 1
        p9 = (idx >> 7) & 1

        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            (p2 == 0 and p3 == 1)
            + (p3 == 0 and p4 == 1)
            + (p4 == 0 and p5 == 1)
            + (p5 == 0 and p6 == 1)
            + (p6 == 0 and p7 == 1)
            + (p7 == 0 and p8 == 1)
            + (p8 == 0 and p9 == 1)
            + (p9 == 0 and p2 == 1)
        )
        if 2 <= b <= 6 and a == 1:
            if (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0):
                lut1[idx] = True
            if (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0):
                lut2[idx] = True

        cond_ne = (p2 == 1 and p4 == 1) and (p6 == 0 and p7 == 0 and p8 == 0)
        cond_nw = (p2 == 1 and p8 == 1) and (p4 == 0 and p5 == 0 and p6 == 0)
        if cond_ne or cond_nw:
            holt1[idx] = True

        cond_se = (p6 == 1 and p4 == 1) and (p2 == 0 and p8 == 0 and p9 == 0)
        cond_sw = (p6 == 1 and p8 == 1) and (p2 == 0 and p3 == 0 and p4 == 0)
        if cond_se or cond_sw:
            holt2[idx] = True

    return lut1, lut2, holt1, holt2


_ZS_LUT1, _ZS_LUT2, _HOLT_LUT1, _HOLT_LUT2 = _build_zhang_suen_luts()


def _zhang_suen_thinning_lut(mask: np.ndarray) -> np.ndarray:
    """Fast pure-NumPy sparse-LUT implementation of Zhang-Suen topological skeletonization.

    Runs in ~4ms on 480x480 masks without Numba, making it compatible with Pyodide/WebAssembly.
    """
    skel = np.pad(mask.astype(np.uint8) > 0, 1, mode="constant").astype(np.uint8)
    ys, xs = np.nonzero(skel[1:-1, 1:-1])
    if len(ys) == 0:
        return np.zeros(mask.shape, dtype=bool)
    ys = ys + 1
    xs = xs + 1

    while len(ys) > 0:
        changed = False
        for lut in (_ZS_LUT1, _ZS_LUT2):
            code = (
                skel[ys - 1, xs]
                | (skel[ys - 1, xs + 1] << 1)
                | (skel[ys, xs + 1] << 2)
                | (skel[ys + 1, xs + 1] << 3)
                | (skel[ys + 1, xs] << 4)
                | (skel[ys + 1, xs - 1] << 5)
                | (skel[ys, xs - 1] << 6)
                | (skel[ys - 1, xs - 1] << 7)
            )
            rem = lut[code]
            if np.any(rem):
                changed = True
                skel[ys[rem], xs[rem]] = 0
                keep = ~rem
                ys = ys[keep]
                xs = xs[keep]
                if len(ys) == 0:
                    break
        if not changed:
            break

    # Post-pass: staircase cleanup (Holt's two-pass L-corner elimination)
    # Ensures strict 8-connected thinness so diagonal strokes do not contain degree-3 staircase pixels
    for holt_lut in (_HOLT_LUT1, _HOLT_LUT2):
        if len(ys) == 0:
            break
        code = (
            skel[ys - 1, xs]
            | (skel[ys - 1, xs + 1] << 1)
            | (skel[ys, xs + 1] << 2)
            | (skel[ys + 1, xs + 1] << 3)
            | (skel[ys + 1, xs] << 4)
            | (skel[ys + 1, xs - 1] << 5)
            | (skel[ys, xs - 1] << 6)
            | (skel[ys - 1, xs - 1] << 7)
        )
        rem = holt_lut[code]
        if np.any(rem):
            skel[ys[rem], xs[rem]] = 0
            keep = ~rem
            ys = ys[keep]
            xs = xs[keep]

    return skel[1:-1, 1:-1] > 0


# Backward-compatible alias
_zhang_suen_thinning_numba = _zhang_suen_thinning_lut


def compute_skeleton(stroke_mask: np.ndarray) -> np.ndarray:
    """Apply Zhang-Suen topological skeletonization to binary stroke mask."""
    if not np.any(stroke_mask):
        return np.zeros_like(stroke_mask, dtype=bool)
    return _zhang_suen_thinning_lut(stroke_mask)


def _get_8_neighbors(skel: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    """Return list of active 8-connected neighbor coordinates (row, col)."""
    h, w = skel.shape
    neighbors: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
                neighbors.append((nr, nc))
    return neighbors


def _compute_degree_map(skel: np.ndarray) -> np.ndarray:
    """Compute 8-connected neighbor count for every foreground pixel in skel."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    counts = ndimage.convolve(skel.astype(np.int32), kernel, mode="constant", cval=0)
    return np.where(skel, counts, 0)


def prune_skeleton_spurs(
    skeleton: np.ndarray,
    edt_radius_px: np.ndarray,
    spur_prune_factor: float = 1.35,
) -> tuple[np.ndarray, int]:
    """Prune terminal spur branches where arc length L(b) <= spur_prune_factor * R(x_J, y_J).

    Traces from degree-1 endpoints up to the first junction node J (deg >= 3).
    Iteratively removes shortest qualifying spurs first so forked tips resolve into
    a single clean spine without deleting the entire stroke.
    """
    skel = skeleton.copy()
    total_pruned = 0

    for _pass in range(4):
        deg_map = _compute_degree_map(skel)
        endpoints_r, endpoints_c = np.where(deg_map == 1)
        if len(endpoints_r) == 0:
            break

        candidate_spurs: list[tuple[float, tuple[int, int], list[tuple[int, int]]]] = []

        for er, ec in zip(endpoints_r, endpoints_c):
            branch_pixels: list[tuple[int, int]] = [(int(er), int(ec))]
            prev = (-1, -1)
            curr = (int(er), int(ec))
            arc_len = 0.0
            junction_node: tuple[int, int] | None = None

            while True:
                nbs = [nb for nb in _get_8_neighbors(skel, curr[0], curr[1]) if nb != prev]
                if len(nbs) == 0:
                    # Reached another endpoint (isolated segment, not a spur on a junction)
                    break

                # Check if any neighbor is a junction (deg >= 3)
                j_nbs = [nb for nb in nbs if deg_map[nb[0], nb[1]] >= 3]
                if j_nbs:
                    junction_node = j_nbs[0]
                    step = np.hypot(junction_node[0] - curr[0], junction_node[1] - curr[1])
                    arc_len += float(step)
                    break

                if len(nbs) > 1:
                    # Current node itself is a junction
                    junction_node = curr
                    break

                nxt = nbs[0]
                step = np.hypot(nxt[0] - curr[0], nxt[1] - curr[1])
                arc_len += float(step)
                if deg_map[nxt[0], nxt[1]] >= 3:
                    junction_node = nxt
                    break

                branch_pixels.append(nxt)
                prev = curr
                curr = nxt

            if junction_node is not None:
                r_j = float(edt_radius_px[junction_node[0], junction_node[1]])
                max_spur_len = spur_prune_factor * r_j
                if arc_len <= max_spur_len:
                    candidate_spurs.append((arc_len, junction_node, branch_pixels))

        if not candidate_spurs:
            break

        # Sort shortest spurs first; at any given junction, prune one spur at a time
        # and re-check junction degree so a 2-way fork at a stroke end leaves 1 main tip
        candidate_spurs.sort(key=lambda item: item[0])
        pruned_in_pass = 0

        for _arc_len, j_node, pixels in candidate_spurs:
            # Verify junction still has degree >= 3 before removing this branch
            j_deg = len(_get_8_neighbors(skel, j_node[0], j_node[1]))
            if j_deg < 3:
                continue
            for pr, pc in pixels:
                skel[pr, pc] = False
            pruned_in_pass += 1
            total_pruned += 1

        if pruned_in_pass == 0:
            break

    return skel, total_pruned


def run_stage2_medial_axis(
    stroke_mask: np.ndarray,
    config: PhysicalCalibrationConfig,
) -> Stage2Result:
    """Execute Stage 2: Euclidean Distance Transform, Skeletonization, and Spur Pruning."""
    # 1. Exact Euclidean Distance Transform (EDT) on M_stroke
    if np.any(stroke_mask):
        edt_radius_px = ndimage.distance_transform_edt(stroke_mask).astype(np.float64)
    else:
        edt_radius_px = np.zeros_like(stroke_mask, dtype=np.float64)

    # 2. Topological Skeletonization
    raw_skeleton = compute_skeleton(stroke_mask)

    # 3. Spur Pruning: Prune(b) <=> L(b) <= 1.35 * R(x_J, y_J)
    pruned_skeleton, pruned_count = prune_skeleton_spurs(
        raw_skeleton,
        edt_radius_px,
        spur_prune_factor=config.spur_prune_factor,
    )

    return Stage2Result(
        edt_radius_px=edt_radius_px,
        raw_skeleton=raw_skeleton,
        pruned_skeleton=pruned_skeleton,
        pruned_branch_count=pruned_count,
    )
