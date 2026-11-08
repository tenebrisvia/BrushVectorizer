"""Stage 5: Polarity Resolution (Start vs. End Detection)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from brush_vectorizer.config import PhysicalCalibrationConfig
from brush_vectorizer.models import PixelPath, PolarityMetrics


@dataclass
class Stage5Result:
    """Outputs of Stage 5 Polarity Resolution."""

    oriented_paths: list[PixelPath]
    reversed_count: int


def _compute_endpoint_steepness(
    s_Dist: np.ndarray,
    radii: np.ndarray,
    window_len: float,
) -> float:
    """Compute max |dr/ds| within the first 15% arc-length window from an endpoint."""
    if len(s_Dist) < 2 or window_len <= 1e-6:
        return 0.0

    mask = s_Dist <= window_len
    if np.count_nonzero(mask) < 2:
        mask = np.arange(min(3, len(s_Dist)))

    s_sub = s_Dist[mask]
    r_sub = radii[mask]
    if len(s_sub) < 2:
        return 0.0

    ds = np.diff(s_sub)
    dr = np.diff(r_sub)
    valid = ds > 1e-5
    if not np.any(valid):
        return 0.0

    # Include boundary step from 0 to r_sub[0] over s=1.0 px to capture blunt landing edge
    blunt_entry_slope = float(r_sub[0]) / max(1.0, float(s_sub[0]) + 1.0)
    internal_slopes = np.abs(dr[valid] / ds[valid])
    return float(max(blunt_entry_slope, float(np.max(internal_slopes))))


def _compute_cross_section_variance(
    coords_xy: np.ndarray,
    radii_px: np.ndarray,
    density_map: np.ndarray,
) -> float:
    """Sample Var(D) perpendicular to the spine across the stroke width (vectorized)."""
    if len(coords_xy) < 2:
        return 0.0

    h, w = density_map.shape
    dx = np.gradient(coords_xy[:, 0])
    dy = np.gradient(coords_xy[:, 1])
    norms = np.hypot(dx, dy)
    norms = np.where(norms > 1e-8, norms, 1.0)
    nx_vec = -dy / norms
    ny_vec = dx / norms

    r_loc = np.maximum(radii_px, 1.5)
    num_samples = max(5, min(15, int(2 * round(float(np.mean(r_loc))) + 1)))
    fracs = np.linspace(-0.85, 0.85, num_samples)[None, :]  # (1, S)
    offsets = r_loc[:, None] * fracs                        # (N, S)

    xs = coords_xy[:, 0:1] + offsets * nx_vec[:, None]
    ys = coords_xy[:, 1:2] + offsets * ny_vec[:, None]
    cols = np.clip(np.round(xs).astype(np.intp), 0, w - 1)
    rows = np.clip(np.round(ys).astype(np.intp), 0, h - 1)
    samples = density_map[rows, cols]                       # (N, S)
    return float(np.mean(np.var(samples, axis=1)))


def evaluate_path_polarity(
    path: PixelPath,
    density_map: np.ndarray,
    config: PhysicalCalibrationConfig,
) -> PolarityMetrics:
    """Compute Start-Likelihood Score for end A (first 15%) and end B (last 15%)."""
    coords = path.coords_xy_px
    radii = path.radii_px
    n_pts = len(coords)

    if n_pts < 3:
        return PolarityMetrics()

    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    s_forward = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(s_forward[-1])
    if total_len <= 1e-5:
        return PolarityMetrics()

    t_norm = s_forward / total_len
    window_len = max(0.15 * total_len, 2.0)

    # 1. Radius Steepness M_steep near A (start) and B (end)
    steep_a = _compute_endpoint_steepness(s_forward, radii, window_len)
    s_backward = total_len - s_forward[::-1]
    steep_b = _compute_endpoint_steepness(s_backward, radii[::-1], window_len)

    # Normalize steepness to comparable [0, 1] range
    steep_denom = max(steep_a + steep_b, 1e-6)
    norm_steep_a = steep_a / steep_denom
    norm_steep_b = steep_b / steep_denom

    # 2. Peak Mass Offset M_mass:
    # Center of mass of r(s) and mean radius in landing zone t in [0.05, 0.25]
    r_sum = float(np.sum(radii))
    if r_sum > 1e-6:
        t_com = float(np.sum(t_norm * radii)) / r_sum
    else:
        t_com = 0.5

    zone_a_mask = (t_norm >= 0.05) & (t_norm <= 0.25)
    zone_b_mask = (t_norm >= 0.75) & (t_norm <= 0.95)
    belly_a = float(np.mean(radii[zone_a_mask])) if np.any(zone_a_mask) else float(radii[0])
    belly_b = float(np.mean(radii[zone_b_mask])) if np.any(zone_b_mask) else float(radii[-1])
    belly_denom = max(belly_a + belly_b, 1e-6)

    mass_a = 0.5 * (1.0 - t_com) + 0.5 * (belly_a / belly_denom)
    mass_b = 0.5 * t_com + 0.5 * (belly_b / belly_denom)

    # 3. Striation Variance M_tex sampled perpendicular to spine in first 15% vs last 15%
    end_a_idx = np.where(t_norm <= 0.15)[0]
    if len(end_a_idx) < 2:
        end_a_idx = np.arange(min(3, n_pts))
    end_b_idx = np.where(t_norm >= 0.85)[0]
    if len(end_b_idx) < 2:
        end_b_idx = np.arange(max(0, n_pts - 3), n_pts)

    tex_a = _compute_cross_section_variance(coords[end_a_idx], radii[end_a_idx], density_map)
    tex_b = _compute_cross_section_variance(coords[end_b_idx], radii[end_b_idx], density_map)
    tex_denom = max(tex_a + tex_b, 1e-6)
    norm_tex_a = tex_a / tex_denom if (tex_a + tex_b) > 1e-6 else 0.0
    norm_tex_b = tex_b / tex_denom if (tex_a + tex_b) > 1e-6 else 0.0

    # Composite score: Score(End) = gamma_1 * M_steep + gamma_2 * M_mass - gamma_3 * M_tex
    score_a = (
        config.gamma_steep * norm_steep_a
        + config.gamma_mass * mass_a
        - config.gamma_tex * norm_tex_a
    )
    score_b = (
        config.gamma_steep * norm_steep_b
        + config.gamma_mass * mass_b
        - config.gamma_tex * norm_tex_b
    )

    reversed_flag = bool(score_b > score_a)
    return PolarityMetrics(
        steepness_a=float(steep_a),
        steepness_b=float(steep_b),
        mass_offset_a=float(mass_a),
        mass_offset_b=float(mass_b),
        tex_variance_a=float(tex_a),
        tex_variance_b=float(tex_b),
        score_a=float(score_a),
        score_b=float(score_b),
        reversed_path=reversed_flag,
    )


def run_stage5_polarity_resolution(
    classified_paths: list[PixelPath],
    density_map: np.ndarray,
    config: PhysicalCalibrationConfig,
) -> Stage5Result:
    """Execute Stage 5: Polarity Resolution (Start vs. End Detection) for all paths."""
    oriented_paths: list[PixelPath] = []
    reversed_count = 0

    for path in classified_paths:
        metrics = evaluate_path_polarity(path, density_map, config)
        if metrics.reversed_path:
            reversed_count += 1
            oriented_paths.append(
                PixelPath(
                    path_id=path.path_id,
                    coords_xy_px=path.coords_xy_px[::-1].copy(),
                    radii_px=path.radii_px[::-1].copy(),
                    densities=path.densities[::-1].copy(),
                    stroke_type=path.stroke_type,
                    polarity_metrics=metrics,
                )
            )
        else:
            oriented_paths.append(
                PixelPath(
                    path_id=path.path_id,
                    coords_xy_px=path.coords_xy_px.copy(),
                    radii_px=path.radii_px.copy(),
                    densities=path.densities.copy(),
                    stroke_type=path.stroke_type,
                    polarity_metrics=metrics,
                )
            )

    return Stage5Result(
        oriented_paths=oriented_paths,
        reversed_count=reversed_count,
    )
