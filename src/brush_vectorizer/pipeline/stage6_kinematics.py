"""Stage 6: Parametric Spline Fitting & Kinematics Compensation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy import interpolate

from brush_vectorizer.config import CanvasTransform, PhysicalCalibrationConfig
from brush_vectorizer.models import PixelPath, Stroke, Waypoint


@dataclass
class ParameterizedStrokeData:
    """Stroke alongside its raw spline centerline (before drag offset) for debug visualization."""

    stroke: Stroke
    raw_spline_xy_mm: np.ndarray
    machine_xy_mm: np.ndarray
    curvature: np.ndarray


@dataclass
class Stage6Result:
    """Outputs of Stage 6 Parametric Spline Fitting & Kinematics Compensation."""

    strokes: list[Stroke]
    debug_data: list[ParameterizedStrokeData]


def parameterize_single_stroke(
    path: PixelPath,
    stroke_id: int,
    transform: CanvasTransform,
    config: PhysicalCalibrationConfig,
) -> ParameterizedStrokeData:
    """Fit 3D cubic B-spline (x(t), y(t), r(t)) and apply kinematics & Z/F profiling."""
    # Convert pixel coordinates and radii to physical canvas millimeters
    xs_mm = path.coords_xy_px[:, 0] * transform.scale_mm_per_px + transform.offset_x_mm
    ys_mm = path.coords_xy_px[:, 1] * transform.scale_mm_per_px + transform.offset_y_mm
    rs_mm = np.clip(
        path.radii_px * transform.scale_mm_per_px,
        0.1,
        config.brush_max_radius_mm,
    )
    densities = np.clip(path.densities, 0.05, 1.0)

    # Compute chord-length parameterization u in [0, 1]
    diffs = np.column_stack([np.diff(xs_mm), np.diff(ys_mm)])
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    keep = np.concatenate([[True], seg_lens > 1e-5])
    xs_mm = xs_mm[keep]
    ys_mm = ys_mm[keep]
    rs_mm = rs_mm[keep]
    densities = densities[keep]

    n_pts = len(xs_mm)
    if n_pts < 2:
        # Degenerate fallback
        xs_mm = np.array([xs_mm[0], xs_mm[0] + 0.1])
        ys_mm = np.array([ys_mm[0], ys_mm[0] + 0.1])
        rs_mm = np.array([rs_mm[0], 0.1])
        densities = np.array([densities[0], densities[0]])
        n_pts = 2

    diffs = np.column_stack([np.diff(xs_mm), np.diff(ys_mm)])
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len_mm = max(float(cum_len[-1]), 1e-4)
    u_chord = cum_len / total_len_mm

    # Number of discrete waypoints N along t in [0, 1]
    num_waypoints = max(
        config.min_waypoints,
        int(math.ceil(total_len_mm / config.waypoint_spacing_mm)) + 1,
    )
    t_vals = np.linspace(0.0, 1.0, num_waypoints)

    # 1. Cubic B-Spline Approximation in 3D space (x(t), y(t), r(t))
    degree = min(3, n_pts - 1)
    # Smoothing factor s proportional to path length and number of points
    s_factor = config.spline_smoothing_factor * n_pts * (transform.scale_mm_per_px ** 2)

    try:
        tck, _u = interpolate.splprep(
            [xs_mm, ys_mm, rs_mm],
            u=u_chord,
            k=degree,
            s=s_factor if n_pts > 4 else 0.0,
        )
        x_t, y_t, r_t = interpolate.splev(t_vals, tck, der=0)
        dx_t, dy_t, _dr_t = interpolate.splev(t_vals, tck, der=1)
        if degree >= 2:
            ddx_t, ddy_t, _ddr_t = interpolate.splev(t_vals, tck, der=2)
        else:
            ddx_t = np.gradient(dx_t, t_vals)
            ddy_t = np.gradient(dy_t, t_vals)

        # Guard against B-spline Runge overshoot on sharp cusps
        margin = 2.5
        if (
            np.min(x_t) < np.min(xs_mm) - margin
            or np.max(x_t) > np.max(xs_mm) + margin
            or np.min(y_t) < np.min(ys_mm) - margin
            or np.max(y_t) > np.max(ys_mm) + margin
        ):
            raise ValueError("Spline overshoot detected; falling back to bounded interpolation.")
    except Exception:
        # Bounded smooth interpolation fallback if spline fitting overshoots or is degenerate
        x_t = np.interp(t_vals, u_chord, xs_mm)
        y_t = np.interp(t_vals, u_chord, ys_mm)
        r_t = np.interp(t_vals, u_chord, rs_mm)
        dx_t = np.gradient(x_t, t_vals)
        dy_t = np.gradient(y_t, t_vals)
        ddx_t = np.gradient(dx_t, t_vals)
        ddy_t = np.gradient(dy_t, t_vals)

    x_t = np.asarray(x_t, dtype=np.float64)
    y_t = np.asarray(y_t, dtype=np.float64)
    r_t = np.clip(np.asarray(r_t, dtype=np.float64), 0.1, config.brush_max_radius_mm)
    dx_t = np.asarray(dx_t, dtype=np.float64)
    dy_t = np.asarray(dy_t, dtype=np.float64)
    ddx_t = np.asarray(ddx_t, dtype=np.float64)
    ddy_t = np.asarray(ddy_t, dtype=np.float64)

    # Interpolate local optical density D(t) along chord length
    d_t = np.clip(np.interp(t_vals, u_chord, densities), 0.05, 1.0)

    # 2. Analytic Curvature Extraction: kappa(t) = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    speed_sq = dx_t * dx_t + dy_t * dy_t
    speed = np.sqrt(np.maximum(speed_sq, 1e-10))
    denom = np.maximum(speed_sq ** 1.5, 1e-10)
    kappa_t = np.abs(dx_t * ddy_t - dy_t * ddx_t) / denom

    # Unit tangent vector T(t)
    tx_t = dx_t / speed
    ty_t = dy_t / speed

    # Taper radius smoothly during exit whip (0.85 <= t <= 1.0) toward 0.1 mm lift
    whip_mask = t_vals >= 0.85
    if np.any(whip_mask):
        whip_phase = (t_vals[whip_mask] - 0.85) / 0.15
        r_whip_start = float(r_t[np.where(whip_mask)[0][0]])
        r_t[whip_mask] = (1.0 - whip_phase) * r_whip_start + whip_phase * min(r_t[-1], 0.1)
        r_t[-1] = 0.1

    # 3. Physical Brush-Drag Offset (Trailer Kinematics):
    # P_machine(t) = [x(t), y(t)]^T + (alpha * r(t)) * T(t)
    alpha = config.brush_drag_coeff
    l_drag = alpha * r_t
    x_machine = x_t + l_drag * tx_t
    y_machine = y_t + l_drag * ty_t

    # 4. Pressure (Z) Mapping:
    # r_eff(t) = r(t) * sqrt(D(t))
    # Z(t) = a * r_eff(t)^2 + b * r_eff(t) + c
    r_eff = r_t * np.sqrt(d_t)
    a_poly, b_poly, c_poly = config.z_calibration_poly
    z_t = a_poly * (r_eff ** 2) + b_poly * r_eff + c_poly

    # 5. Velocity (F) Profiling & Exit Whip Z Lift:
    # - Entry Press (0 <= t <= 0.12): clamp feedrate to F_min
    # - Mid-Stroke (0.12 < t < 0.85): F(t) = max(F_min, F_cruise / (1 + k_kappa * kappa(t)))
    # - Exit Whip (0.85 <= t <= 1.0): linearly ramp feedrate toward F_max while lifting Z(t) -> Z_clearance
    feed_t = np.empty_like(t_vals)
    entry_mask = t_vals <= 0.12
    mid_mask = (t_vals > 0.12) & (t_vals < 0.85)

    feed_t[entry_mask] = config.f_min_mm_min
    feed_t[mid_mask] = np.maximum(
        config.f_min_mm_min,
        config.f_cruise_mm_min / (1.0 + config.curvature_slowdown_gain * kappa_t[mid_mask]),
    )

    if np.any(whip_mask):
        whip_phase = (t_vals[whip_mask] - 0.85) / 0.15
        # Continuity from end of mid-stroke
        mid_indices = np.where(mid_mask)[0]
        f_whip_start = float(feed_t[mid_indices[-1]]) if len(mid_indices) > 0 else config.f_cruise_mm_min
        feed_t[whip_mask] = (1.0 - whip_phase) * f_whip_start + whip_phase * config.f_max_mm_min

        # Lift Z(t) -> Z_clearance smoothly during Exit Whip
        z_whip_start = float(z_t[mid_indices[-1]]) if len(mid_indices) > 0 else float(z_t[0])
        z_t[whip_mask] = (1.0 - whip_phase) * z_whip_start + whip_phase * config.z_clearance_mm
        z_t[-1] = config.z_clearance_mm

    # Compute dwell_start_ms in [30, 60] ms scaled by initial effective radius ratio
    init_ratio = float(np.clip(r_eff[0] / max(config.brush_max_radius_mm, 1e-3), 0.0, 1.0))
    dwell_start_ms = int(round(30.0 + 30.0 * init_ratio))

    waypoints = [
        Waypoint(
            t=round(float(t_vals[i]), 4),
            x_mm=round(float(x_machine[i]), 3),
            y_mm=round(float(y_machine[i]), 3),
            r_mm=round(float(r_t[i]), 3),
            z_mm=round(float(z_t[i]), 3),
            feed_mm_min=round(float(feed_t[i]), 1),
        )
        for i in range(num_waypoints)
    ]

    stroke = Stroke(
        stroke_id=stroke_id,
        stroke_type=path.stroke_type,
        dwell_start_ms=dwell_start_ms,
        waypoints=waypoints,
    )

    return ParameterizedStrokeData(
        stroke=stroke,
        raw_spline_xy_mm=np.column_stack([x_t, y_t]),
        machine_xy_mm=np.column_stack([x_machine, y_machine]),
        curvature=kappa_t,
    )


def run_stage6_kinematics(
    oriented_paths: list[PixelPath],
    transform: CanvasTransform,
    config: PhysicalCalibrationConfig,
) -> Stage6Result:
    """Execute Stage 6: Parametric Spline Fitting & Kinematics Compensation."""
    strokes: list[Stroke] = []
    debug_data: list[ParameterizedStrokeData] = []

    for idx, path in enumerate(oriented_paths):
        pdata = parameterize_single_stroke(path, idx, transform, config)
        strokes.append(pdata.stroke)
        debug_data.append(pdata)

    return Stage6Result(strokes=strokes, debug_data=debug_data)
