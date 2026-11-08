"""Section 4: Global Path Ordering (Plotter Optimization)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from brush_vectorizer.config import PhysicalCalibrationConfig
from brush_vectorizer.models import Stroke
from brush_vectorizer.pipeline.stage6_kinematics import ParameterizedStrokeData


@dataclass
class OrderingResult:
    """Outputs of Global Path Ordering."""

    ordered_strokes: list[Stroke]
    ordered_debug_data: list[ParameterizedStrokeData]
    air_travel_before_mm: float
    air_travel_after_mm: float


def _stroke_start_xy(stroke: Stroke) -> tuple[float, float]:
    wp = stroke.waypoints[0]
    return (wp.x_mm, wp.y_mm)


def _stroke_end_xy(stroke: Stroke) -> tuple[float, float]:
    wp = stroke.waypoints[-1]
    return (wp.x_mm, wp.y_mm)


def _stroke_mean_y(stroke: Stroke) -> float:
    return float(np.mean([wp.y_mm for wp in stroke.waypoints]))


def _compute_air_travel(strokes: list[Stroke]) -> float:
    """Compute total non-drawing carriage travel distance (mm) starting from origin (0, 0)."""
    if not strokes:
        return 0.0
    total = 0.0
    curr_x, curr_y = 0.0, 0.0
    for st in strokes:
        sx, sy = _stroke_start_xy(st)
        total += math.hypot(sx - curr_x, sy - curr_y)
        curr_x, curr_y = _stroke_end_xy(st)
    return total


def _solve_band_tsp(
    band_items: list[ParameterizedStrokeData],
    start_pos: tuple[float, float],
) -> list[ParameterizedStrokeData]:
    """Order strokes within a spatial Y-band using Nearest-Neighbor + directed 2-opt TSP.

    Preserves each stroke's internal polarity (start -> end) determined in Stage 5.
    """
    n = len(band_items)
    if n <= 1:
        return list(band_items)

    # 1. Greedy Nearest-Neighbor initialization from start_pos
    unvisited = list(range(n))
    order: list[int] = []
    curr_x, curr_y = start_pos

    while unvisited:
        best_idx = -1
        best_dist = float("inf")
        for idx in unvisited:
            sx, sy = _stroke_start_xy(band_items[idx].stroke)
            # Slight bias toward smaller Y to respect top-to-bottom wet-ink avoidance inside the band
            dy_bias = 0.15 * max(0.0, sy - curr_y)
            d = math.hypot(sx - curr_x, sy - curr_y) + dy_bias
            if d < best_dist:
                best_dist = d
                best_idx = idx
        order.append(best_idx)
        unvisited.remove(best_idx)
        curr_x, curr_y = _stroke_end_xy(band_items[best_idx].stroke)

    # 2. Directed 2-opt local search (swapping pairs/subsequences without reversing stroke polarity)
    def _seq_cost(seq: list[int]) -> float:
        cx, cy = start_pos
        cost = 0.0
        for idx in seq:
            st = band_items[idx].stroke
            sx, sy = _stroke_start_xy(st)
            cost += math.hypot(sx - cx, sy - cy)
            cx, cy = _stroke_end_xy(st)
        return cost

    best_cost = _seq_cost(order)
    if n <= 60:
        improved = True
        iters = 0
        while improved and iters < 25:
            improved = False
            iters += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = order.copy()
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    c_cost = _seq_cost(candidate)
                    if c_cost + 1e-3 < best_cost:
                        order = candidate
                        best_cost = c_cost
                        improved = True

    return [band_items[idx] for idx in order]


def run_global_path_ordering(
    debug_data: list[ParameterizedStrokeData],
    config: PhysicalCalibrationConfig,
) -> OrderingResult:
    """Execute Section 4 Global Path Ordering: Broad-to-Fine Layering + Banded TSP."""
    if not debug_data:
        return OrderingResult(
            ordered_strokes=[],
            ordered_debug_data=[],
            air_travel_before_mm=0.0,
            air_travel_after_mm=0.0,
        )

    air_before = _compute_air_travel([d.stroke for d in debug_data])

    # 1. Layering (Broad to Fine): charcoal_wash first, calligraphic_centerline second
    wash_layer = [d for d in debug_data if d.stroke.stroke_type == "charcoal_wash"]
    calligraphic_layer = [
        d for d in debug_data if d.stroke.stroke_type == "calligraphic_centerline"
    ]

    canvas_h = config.canvas_dims_mm[1]
    num_bands = max(1, config.num_spatial_bands)
    band_height = max(canvas_h / num_bands, 1.0)

    ordered_debug: list[ParameterizedStrokeData] = []
    curr_pos = (0.0, 0.0)

    # 2. Spatial Sweeping (Top-to-Bottom Y bands + intra-band TSP)
    for layer_items in (wash_layer, calligraphic_layer):
        if not layer_items:
            continue

        bands: dict[int, list[ParameterizedStrokeData]] = {b: [] for b in range(num_bands)}
        for item in layer_items:
            mean_y = _stroke_mean_y(item.stroke)
            b_idx = int(np.clip(math.floor(mean_y / band_height), 0, num_bands - 1))
            bands[b_idx].append(item)

        for b_idx in range(num_bands):
            if not bands[b_idx]:
                continue
            band_ordered = _solve_band_tsp(bands[b_idx], curr_pos)
            ordered_debug.extend(band_ordered)
            curr_pos = _stroke_end_xy(band_ordered[-1].stroke)

    # Re-index stroke_id sequentially in final execution order
    ordered_strokes: list[Stroke] = []
    for new_id, item in enumerate(ordered_debug):
        item.stroke.stroke_id = new_id
        ordered_strokes.append(item.stroke)

    air_after = _compute_air_travel(ordered_strokes)

    return OrderingResult(
        ordered_strokes=ordered_strokes,
        ordered_debug_data=ordered_debug,
        air_travel_before_mm=round(air_before, 2),
        air_travel_after_mm=round(air_after, 2),
    )
