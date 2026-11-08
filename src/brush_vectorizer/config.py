"""Configuration and physical calibration schemas for the vectorization pipeline."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PhysicalCalibrationConfig(BaseModel):
    """Physical Calibration Config (C) as defined in Section 2.1 of the specification."""

    # Physical target dimensions (W_mm, H_mm)
    canvas_dims_mm: tuple[float, float] = Field(
        default=(210.0, 297.0),
        description="(W_mm, H_mm) physical target dimensions in millimeters (default: A4).",
    )
    # Physical splay limit of the target brush (R_max)
    brush_max_radius_mm: float = Field(
        default=3.5,
        gt=0.0,
        description="R_max: the physical splay limit of the target brush in millimeters.",
    )
    # Empirical hitch-length constant for bristle lag compensation (alpha)
    brush_drag_coeff: float = Field(
        default=0.85,
        ge=0.0,
        description="alpha: empirical hitch-length constant for bristle lag compensation.",
    )
    # Coefficients [a, b, c] mapping target effective radius r to physical plotter height Z
    z_calibration_poly: tuple[float, float, float] = Field(
        default=(-0.15, -1.10, 0.0),
        description="[a, b, c] polynomial coefficients: Z(t) = a * r_eff^2 + b * r_eff + c.",
    )

    # Feedrate & Z clearance parameters for Stage 6
    f_min_mm_min: float = Field(
        default=400.0,
        gt=0.0,
        description="Minimum feedrate F_min (mm/min) used during entry press and tight turns.",
    )
    f_cruise_mm_min: float = Field(
        default=1800.0,
        gt=0.0,
        description="Nominal cruising feedrate F_cruise (mm/min) during mid-stroke.",
    )
    f_max_mm_min: float = Field(
        default=3000.0,
        gt=0.0,
        description="Maximum feedrate F_max (mm/min) reached at exit whip.",
    )
    z_clearance_mm: float = Field(
        default=0.0,
        description="Z_clearance height (mm) at stroke completion / exit whip.",
    )
    curvature_slowdown_gain: float = Field(
        default=5.0,
        ge=0.0,
        description="k_kappa: curvature gain controlling slowdown around tight bends.",
    )

    # Stage 1 parameters
    anisotropic_sigma: float = Field(
        default=1.0,
        ge=0.0,
        description="Gaussian / anisotropic bridging smoothing scale in pixels.",
    )
    threshold_mode: Literal["otsu", "fixed", "hybrid"] = Field(
        default="hybrid",
        description="Thresholding method: 'otsu', 'fixed' (density_floor), or 'hybrid' (min of otsu and floor).",
    )
    density_floor: float = Field(
        default=0.15,
        ge=0.01,
        le=0.99,
        description="Fixed optical density floor (D > 0.15).",
    )

    # Stage 2 parameters
    spur_prune_factor: float = Field(
        default=1.35,
        ge=0.0,
        description="Prune terminal branches with L(b) <= spur_prune_factor * R(x_J, y_J).",
    )
    min_stroke_length_px: float = Field(
        default=4.0,
        ge=1.0,
        description="Minimum skeleton polyline length in pixels to retain.",
    )

    # Stage 3 parameters
    bridge_collapse_factor: float = Field(
        default=2.0,
        ge=0.0,
        description="Collapse junction-to-junction bridges shorter than factor * R_J.",
    )
    weight_theta: float = Field(default=0.65, ge=0.0)
    weight_radius: float = Field(default=0.25, ge=0.0)
    weight_density: float = Field(default=0.10, ge=0.0)
    max_deflection_deg: float = Field(
        default=65.0,
        gt=0.0,
        le=180.0,
        description="Maximum allowed deflection angle in degrees for greedy path merging.",
    )

    # Stage 4 parameters
    wash_density_threshold: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description="Mean optical density threshold below which wide strokes become charcoal wash.",
    )
    wash_Stepover_factor: float = Field(
        default=0.70,
        gt=0.1,
        le=2.0,
        description="Spacing factor (0.7 * R_max) for multi-pass charcoal wash offset curves.",
    )

    # Stage 5 polarity weights
    gamma_steep: float = Field(default=0.45, ge=0.0)
    gamma_mass: float = Field(default=0.35, ge=0.0)
    gamma_tex: float = Field(default=0.20, ge=0.0)

    # Stage 6 sampling & spline smoothing
    spline_smoothing_factor: float = Field(
        default=0.35,
        ge=0.0,
        description="Smoothing factor multiplier proportional to path length for cubic B-spline.",
    )
    waypoint_spacing_mm: float = Field(
        default=0.75,
        gt=0.05,
        description="Approximate spacing in mm between sampled waypoints along each stroke.",
    )
    min_waypoints: int = Field(
        default=12,
        ge=4,
        description="Minimum number of waypoints sampled per stroke.",
    )

    # Section 4 Global Path Ordering
    num_spatial_bands: int = Field(
        default=8,
        ge=1,
        description="Number of Y-axis spatial sweeping bands for wet-ink avoidance + TSP.",
    )


class CanvasTransform(BaseModel):
    """Uniform coordinate transform from image pixels to physical canvas millimeters."""

    image_width_px: int
    image_height_px: int
    canvas_width_mm: float
    canvas_height_mm: float
    scale_mm_per_px: float
    offset_x_mm: float
    offset_y_mm: float

    @classmethod
    def from_image_and_canvas(
        cls, width_px: int, height_px: int, canvas_dims_mm: tuple[float, float]
    ) -> "CanvasTransform":
        """Create a uniform scaling transform preserving image orientation and centering on canvas."""
        w_mm, h_mm = canvas_dims_mm
        scale = min(w_mm / max(width_px, 1), h_mm / max(height_px, 1))
        offset_x = (w_mm - width_px * scale) / 2.0
        offset_y = (h_mm - height_px * scale) / 2.0
        return cls(
            image_width_px=width_px,
            image_height_px=height_px,
            canvas_width_mm=w_mm,
            canvas_height_mm=h_mm,
            scale_mm_per_px=scale,
            offset_x_mm=offset_x,
            offset_y_mm=offset_y,
        )

    def px_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert pixel coordinates (x=col, y=row) to physical mm (same top-left orientation)."""
        return (
            x_px * self.scale_mm_per_px + self.offset_x_mm,
            y_px * self.scale_mm_per_px + self.offset_y_mm,
        )

    def radius_px_to_mm(self, r_px: float) -> float:
        """Convert a pixel distance/radius to physical millimeters."""
        return r_px * self.scale_mm_per_px

    def radius_mm_to_px(self, r_mm: float) -> float:
        """Convert a physical millimeter radius to pixel units."""
        return r_mm / self.scale_mm_per_px


class PainterlyConfig(BaseModel):
    """Configuration for Monitor-Targeted Multi-Layer Painterly Vectorization."""

    num_tonal_layers: int = Field(
        default=7,
        ge=2,
        le=16,
        description="Number of discrete color/tonal layers restricted from the source image.",
    )
    color_palette_mode: Literal["auto_palette", "monochrome"] = Field(
        default="auto_palette",
        description="'auto_palette' extracts K dominant colors from the source image; 'monochrome' uses grayscale tones.",
    )
    layer_colors: list[str | None] | None = Field(
        default=None,
        description="Optional per-layer hex color overrides (e.g. ['#e2c4a8', None, '#2b1d18']).",
    )
    layer_blurs: list[float | None] | None = Field(
        default=None,
        description="Optional per-layer Gaussian blur stdDeviation overrides in pixels (0.0 to 12.0).",
    )
    layer_thresholds: list[float] | None = Field(
        default=None,
        description="Optional explicit optical density thresholds [d_1, ..., d_K] in (0, 1). If None, computed adaptively.",
    )
    min_density_threshold: float = Field(
        default=0.10,
        ge=0.02,
        le=0.80,
        description="Lowest optical density threshold for the faintest wash layer.",
    )
    max_density_threshold: float = Field(
        default=0.84,
        ge=0.20,
        le=0.98,
        description="Highest optical density threshold for the darkest accent layer.",
    )
    layer_opacity_range: tuple[float, float] = Field(
        default=(0.28, 0.94),
        description="(min_opacity, max_opacity) across tonal layers from faintest wash to deepest black.",
    )
    bristle_count: int = Field(
        default=5,
        ge=0,
        le=16,
        description="Number of dry-brush bristle sub-filaments generated per stroke.",
    )
    dry_brush_strength: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Intensity of bristle striation separation and paper-tooth texture.",
    )
    paper_grain_strength: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="Strength of subtle paper grain texture on the canvas background.",
    )
    match_source_paper_tone: bool = Field(
        default=True,
        description="Sample the natural paper color/tone from the source image background.",
    )
    fallback_paper_hex: str = Field(
        default="#f4f1ea",
        description="Default warm paper hex color if match_source_paper_tone is False.",
    )
    normalize_paper_illumination: bool = Field(
        default=True,
        description="Normalize uneven lighting across the paper while preserving tonal washes.",
    )
    anisotropic_sigma: float = Field(
        default=1.2,
        ge=0.0,
        description="Structure-tensor anisotropic smoothing scale (px) for bridging bristle gaps.",
    )
    spur_prune_factor: float = Field(
        default=1.25,
        ge=0.0,
        description="Spur pruning ratio L(b) <= factor * R_J.",
    )
    min_stroke_length_px: float = Field(
        default=6.0,
        ge=2.0,
        description="Minimum skeleton polyline length in pixels.",
    )
    max_deflection_deg: float = Field(
        default=65.0,
        gt=0.0,
        le=180.0,
        description="Maximum deflection angle at junctions for continuous stroke merging.",
    )
    spline_smoothing_factor: float = Field(
        default=0.40,
        ge=0.0,
        description="B-spline smoothing factor multiplier.",
    )
    stroke_width_scale: float = Field(
        default=1.12,
        gt=0.1,
        le=3.0,
        description="Multiplier on EDT radius to ensure overlapping washes blend seamlessly.",
    )

    # Creative & Abstract Effects Parameters
    nib_angle_deg: float = Field(
        default=35.0,
        ge=0.0,
        le=180.0,
        description="Angle of calligraphic chisel/palette-knife nib in degrees.",
    )
    nib_flatness: float = Field(
        default=0.0,
        ge=0.0,
        le=0.95,
        description="0.0 = round brush; 0.95 = razor-thin calligraphic chisel nib.",
    )
    taper_power: float = Field(
        default=1.0,
        ge=-2.0,
        le=4.0,
        description="End-cap tapering profile: 0 = blunt tape, 1 = natural, >1.5 = comet whip, <0 = inverted teardrop.",
    )
    radius_gamma: float = Field(
        default=1.0,
        ge=-1.5,
        le=2.5,
        description="Power-law exponent on stroke width: 1 = natural, 0 = monoline contour, <0 = inverted widths.",
    )
    polygon_decimation: float = Field(
        default=0.0,
        ge=0.0,
        le=0.95,
        description="0.0 = smooth curves; higher = decimates strokes into low-poly origami facets.",
    )
    spine_wave_amplitude: float = Field(
        default=0.0,
        ge=0.0,
        le=40.0,
        description="Perpendicular sine-wave oscillation amplitude along each stroke spine (px).",
    )
    spine_wave_frequency: float = Field(
        default=2.0,
        ge=0.25,
        le=16.0,
        description="Number of wave cycles along each stroke spine.",
    )
    bristle_splay: float = Field(
        default=0.14,
        ge=0.0,
        le=4.0,
        description="How far bristle filaments fan out at the tail of each stroke.",
    )
    stroke_extension_px: float = Field(
        default=0.0,
        ge=0.0,
        le=60.0,
        description="Extrapolates start/end tangents past stroke endpoints (px) for sketchy overshoots.",
    )
    blend_mode: Literal[
        "normal",
        "multiply",
        "screen",
        "overlay",
        "difference",
        "exclusion",
        "color-dodge",
        "hard-light",
    ] = Field(
        default="normal",
        description="CSS mix-blend-mode applied to painted SVG layers.",
    )
    turbulence_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=12.0,
        description="Multiplier on SVG feDisplacementMap turbulence warp for wet-ink edge bleed.",
    )
    stroke_outline_mode: Literal["filled", "outlined", "hollow"] = Field(
        default="filled",
        description="'filled' = solid paint; 'outlined' = paint + keyline border; 'hollow' = contour wireframe.",
    )
    stroke_outline_width: float = Field(
        default=1.2,
        ge=0.2,
        le=8.0,
        description="Outline stroke width in pixels when stroke_outline_mode is 'outlined' or 'hollow'.",
    )
    stroke_hue_jitter: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Impressionist broken-color shimmer: randomly shifts individual stroke hues around the layer color.",
    )
    layer_misalignment_px: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description="Misregisters tonal layers by deterministic (dx, dy) pixel shifts (Risograph effect).",
    )
    flow_lock_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overrides natural image gradient flow with a forced directional closing angle.",
    )
    flow_lock_angle_deg: float = Field(
        default=45.0,
        ge=0.0,
        le=180.0,
        description="Forced directional bridging angle in degrees when flow_lock_strength > 0.",
    )

