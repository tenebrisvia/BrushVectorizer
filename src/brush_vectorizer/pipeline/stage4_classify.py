"""Stage 4: Stroke Classification (Calligraphic vs. Broad Wash)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage

from brush_vectorizer.config import CanvasTransform, PhysicalCalibrationConfig
from brush_vectorizer.models import PixelPath


@dataclass
class Stage4Result:
    """Outputs of Stage 4 Stroke Classification & Multi-Pass Sub-division."""

    classified_paths: list[PixelPath]
    calligraphic_count: int
    wash_region_count: int
    wash_subpass_count: int


def _compute_smooth_normals(coords_xy: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Compute smoothed unit normal vectors along a 2D polyline."""
    if len(coords_xy) < 2:
        return np.zeros_like(coords_xy)

    xs = ndimage.gaussian_filter1d(coords_xy[:, 0], sigma=min(sigma, max(0.5, len(coords_xy) / 4)))
    ys = ndimage.gaussian_filter1d(coords_xy[:, 1], sigma=min(sigma, max(0.5, len(coords_xy) / 4)))

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    norms = np.hypot(dx, dy)
    norms = np.where(norms > 1e-8, norms, 1.0)
    tx = dx / norms
    ty = dy / norms

    # Unit normal (-ty, tx)
    nx_vec = -ty
    ny_vec = tx
    return np.column_stack([nx_vec, ny_vec])


def _subdivide_broad_wash(
    path: PixelPath,
    stroke_mask: np.ndarray,
    edt_radius_px: np.ndarray,
    density_map: np.ndarray,
    stepover_px: float,
    r_max_px: float,
    min_length_px: float,
    start_id: int,
) -> list[PixelPath]:
    """Generate parallel offset curves spaced by 0.7 * R_max bounded by the original mask contour."""
    coords = path.coords_xy_px
    # Smooth the base spine coordinates before offsetting to prevent normal cusps at tight corners
    smooth_sigma = min(3.0, max(0.5, len(coords) / 6.0))
    base_x = ndimage.gaussian_filter1d(coords[:, 0], sigma=smooth_sigma)
    base_y = ndimage.gaussian_filter1d(coords[:, 1], sigma=smooth_sigma)
    base_coords = np.column_stack([base_x, base_y])
    normals = _compute_smooth_normals(base_coords, sigma=3.0)
    max_r_px = float(np.max(path.radii_px))
    h, w = stroke_mask.shape

    num_offsets_each_side = max(1, int(math.ceil((max_r_px - 0.5 * r_max_px) / max(stepover_px, 1.0))))
    offsets = [0.0]
    for k in range(1, num_offsets_each_side + 1):
        offsets.append(k * stepover_px)
        offsets.append(-k * stepover_px)
    offsets.sort()

    sub_paths: list[PixelPath] = []
    next_id = start_id

    for offset_px in offsets:
        offset_coords = base_coords + offset_px * normals

        # Check which points lie inside the original stroke_mask contour and within local stroke width
        cols = np.round(offset_coords[:, 0]).astype(int)
        rows = np.round(offset_coords[:, 1]).astype(int)
        in_bounds = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)

        valid = np.zeros(len(offset_coords), dtype=bool)
        valid_indices = np.where(in_bounds)[0]
        if len(valid_indices) > 0:
            v_rows = rows[valid_indices]
            v_cols = cols[valid_indices]
            inside_mask = stroke_mask[v_rows, v_cols]
            within_width = np.abs(offset_px) <= (path.radii_px[valid_indices] + 0.5)
            valid[valid_indices] = inside_mask & within_width

        # Split into contiguous valid runs
        padded = np.concatenate([[False], valid, [False]])
        starts = np.where(padded[1:] & ~padded[:-1])[0]
        ends = np.where(~padded[1:] & padded[:-1])[0]

        for s_idx, e_idx in zip(starts, ends):
            if e_idx - s_idx < 2:
                continue
            seg_coords = offset_coords[s_idx:e_idx]
            diffs = np.diff(seg_coords, axis=0)
            seg_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
            if seg_len < min_length_px:
                continue

            seg_cols = np.clip(np.round(seg_coords[:, 0]).astype(int), 0, w - 1)
            seg_rows = np.clip(np.round(seg_coords[:, 1]).astype(int), 0, h - 1)

            local_edt = edt_radius_px[seg_rows, seg_cols].astype(np.float64)
            # Cap sub-pass radius to physical brush max radius
            seg_radii = np.clip(np.minimum(local_edt, r_max_px), 0.5, r_max_px)
            seg_densities = np.clip(density_map[seg_rows, seg_cols].astype(np.float64), 0.05, 1.0)

            sub_paths.append(
                PixelPath(
                    path_id=next_id,
                    coords_xy_px=seg_coords,
                    radii_px=seg_radii,
                    densities=seg_densities,
                    stroke_type="charcoal_wash",
                )
            )
            next_id += 1

    return sub_paths


def run_stage4_classification(
    untangled_paths: list[PixelPath],
    stroke_mask: np.ndarray,
    edt_radius_px: np.ndarray,
    density_map: np.ndarray,
    transform: CanvasTransform,
    config: PhysicalCalibrationConfig,
) -> Stage4Result:
    """Execute Stage 4: Classify strokes into Calligraphic vs. Charcoal Wash & subdivide."""
    r_max_mm = config.brush_max_radius_mm
    r_max_px = transform.radius_mm_to_px(r_max_mm)
    stepover_px = transform.radius_mm_to_px(config.wash_Stepover_factor * r_max_mm)

    classified_paths: list[PixelPath] = []
    calligraphic_count = 0
    wash_region_count = 0
    wash_subpass_count = 0

    for path in untangled_paths:
        max_r_mm = transform.radius_px_to_mm(float(np.max(path.radii_px)))
        mean_density = float(np.mean(path.densities))

        # Per Stage 4 spec:
        # Charcoal Wash / Broad Shading: max(r_i) > R_max and mean optical density D_bar <= wash_density_threshold
        if max_r_mm > r_max_mm and mean_density <= config.wash_density_threshold:
            wash_region_count += 1
            sub_paths = _subdivide_broad_wash(
                path=path,
                stroke_mask=stroke_mask,
                edt_radius_px=edt_radius_px,
                density_map=density_map,
                stepover_px=stepover_px,
                r_max_px=r_max_px,
                min_length_px=config.min_stroke_length_px,
                start_id=len(classified_paths),
            )
            if sub_paths:
                classified_paths.extend(sub_paths)
                wash_subpass_count += len(sub_paths)
            else:
                # Fallback if region is too short to split
                fallback = PixelPath(
                    path_id=len(classified_paths),
                    coords_xy_px=path.coords_xy_px.copy(),
                    radii_px=np.minimum(path.radii_px, r_max_px),
                    densities=path.densities.copy(),
                    stroke_type="charcoal_wash",
                )
                classified_paths.append(fallback)
                wash_subpass_count += 1
        else:
            # Calligraphic Stroke: single centerline stroke
            calligraphic_path = PixelPath(
                path_id=len(classified_paths),
                coords_xy_px=path.coords_xy_px.copy(),
                radii_px=np.minimum(path.radii_px, r_max_px),
                densities=path.densities.copy(),
                stroke_type="calligraphic_centerline",
            )
            classified_paths.append(calligraphic_path)
            calligraphic_count += 1

    return Stage4Result(
        classified_paths=classified_paths,
        calligraphic_count=calligraphic_count,
        wash_region_count=wash_region_count,
        wash_subpass_count=wash_subpass_count,
    )
