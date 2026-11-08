"""Data contracts and intermediate representations for the vectorization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field


StrokeType = Literal["calligraphic_centerline", "charcoal_wash"]


class Waypoint(BaseModel):
    """Discrete waypoint along parameter t in [0, 1] as specified in Section 2.2."""

    t: float = Field(..., description="Normalized parameter t in [0.0, 1.0].")
    x_mm: float = Field(..., description="Physical X machine coordinate in mm.")
    y_mm: float = Field(..., description="Physical Y machine coordinate in mm.")
    r_mm: float = Field(..., description="Target stroke radius in mm.")
    z_mm: float = Field(..., description="Physical plotter Z height in mm.")
    feed_mm_min: float = Field(..., description="Target feedrate in mm/min.")


class Stroke(BaseModel):
    """Single vectorized brush stroke S_k as specified in Section 2.2."""

    stroke_id: int = Field(..., description="Unique ordered stroke identifier.")
    stroke_type: StrokeType = Field(
        default="calligraphic_centerline",
        description="Stroke classification ('calligraphic_centerline' or 'charcoal_wash').",
    )
    layer_index: int = Field(
        default=0,
        description="0-based tonal/color layer index (0 = lightest background wash, K-1 = darkest accent).",
    )
    layer_name: str = Field(
        default="",
        description="Layer identifier (e.g. '1_faint_wash') for multi-layer plotter output.",
    )
    color_hex: str = Field(
        default="#111827",
        description="Active hex paint color (#rrggbb) for this stroke.",
    )
    dwell_start_ms: int = Field(
        default=45,
        description="Initial bristle compression dwell time at t=0 in milliseconds.",
    )
    waypoints: list[Waypoint] = Field(
        default_factory=list,
        description="Ordered sequence of discretely sampled waypoints along t in [0, 1].",
    )


class PlotterLayerInfo(BaseModel):
    """Metadata for a single color/tonal paint layer in a multi-layer Plotter document."""

    layer_index: int = Field(..., description="0-based layer index (0 = lightest wash, K-1 = darkest accent).")
    layer_name: str = Field(..., description="Inkscape layer name, e.g. '1_faint_wash'.")
    density_threshold: float = Field(..., description="Minimum optical density threshold D_k for this layer.")
    target_gray: float = Field(..., description="Characteristic luminance in [0.0, 1.0].")
    color_hex: str = Field(..., description="Active paint color hex (#rrggbb).")
    auto_color_hex: str = Field(..., description="Extracted source palette color hex (#rrggbb).")
    stroke_count: int = Field(..., description="Number of centerline strokes in this layer.")
    air_travel_mm: float = Field(default=0.0, description="Optimized non-drawing rapid travel distance (mm) within this layer.")


class StrokeVectorDocument(BaseModel):
    """Top-level structured Stroke Vector Document."""

    canvas_dims_mm: tuple[float, float]
    source_dims_px: tuple[int, int]
    stroke_count: int
    strokes: list[Stroke]
    paper_hex: str = Field(default="#f4f1ea", description="Background paper canvas hex color.")
    layers: list[PlotterLayerInfo] = Field(
        default_factory=list,
        description="Optional multi-layer paint metadata when generated via the multi-layer plotter pipeline.",
    )
    air_travel_before_mm: float = Field(default=0.0, description="Total rapid travel before TSP ordering (mm).")
    air_travel_after_mm: float = Field(default=0.0, description="Total rapid travel after per-layer TSP ordering (mm).")


@dataclass
class PolarityMetrics:
    """Diagnostic polarity scores for ends A (first 15%) and B (last 15%)."""

    steepness_a: float = 0.0
    steepness_b: float = 0.0
    mass_offset_a: float = 0.0
    mass_offset_b: float = 0.0
    tex_variance_a: float = 0.0
    tex_variance_b: float = 0.0
    score_a: float = 0.0
    score_b: float = 0.0
    reversed_path: bool = False


@dataclass
class PixelPath:
    """Continuous polyline in image pixel space before spline parameterization."""

    path_id: int
    # Ordered pixel coordinates shape (N, 2) where columns are (x_px, y_px) = (col, row)
    coords_xy_px: np.ndarray
    # Local EDT radius in pixels at each point, shape (N,)
    radii_px: np.ndarray
    # Local optical density D(x, y) in [0, 1] at each point, shape (N,)
    densities: np.ndarray
    stroke_type: StrokeType = "calligraphic_centerline"
    polarity_metrics: PolarityMetrics | None = field(default=None)

    @property
    def arc_length_px(self) -> float:
        """Compute total polyline arc length in pixels."""
        if len(self.coords_xy_px) < 2:
            return 0.0
        diffs = np.diff(self.coords_xy_px, axis=0)
        return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
