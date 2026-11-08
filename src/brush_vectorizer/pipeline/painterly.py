"""Multi-layer color & tonal painterly vectorization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import colorsys
import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import cluster, interpolate, ndimage

from brush_vectorizer.config import CanvasTransform, PainterlyConfig, PhysicalCalibrationConfig
from brush_vectorizer.models import (
    PixelPath,
    PlotterLayerInfo,
    Stroke,
    StrokeVectorDocument,
    Waypoint,
)
from brush_vectorizer.pipeline.ordering import _compute_air_travel, _solve_band_tsp, _stroke_end_xy
from brush_vectorizer.pipeline.stage1_preprocess import (
    anisotropic_gap_bridging,
    load_luminance_image,
    normalize_paper_luminance,
)
from brush_vectorizer.pipeline.stage2_medial_axis import run_stage2_medial_axis
from brush_vectorizer.pipeline.stage3_graph import run_stage3_graph_untangling
from brush_vectorizer.pipeline.stage4_classify import _subdivide_broad_wash
from brush_vectorizer.pipeline.stage5_polarity import run_stage5_polarity_resolution
from brush_vectorizer.pipeline.stage6_kinematics import ParameterizedStrokeData


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (int(np.clip(c, 0, 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        return (64, 64, 64)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _modulate_rgb_tone(
    base_rgb: tuple[int, int, int],
    paper_rgb: tuple[int, int, int],
    shade_factor: float,
) -> tuple[int, int, int]:
    """Modulate a layer's base RGB color by shade_factor (>1 darker press, <1 lighter toward paper)."""
    r, g, b = base_rgb
    pr, pg, pb = paper_rgb
    if shade_factor >= 1.0:
        # Darken slightly for wet press or bristle core
        dark_mult = max(0.25, 1.0 / shade_factor)
        return (
            int(np.clip(round(r * dark_mult), 0, 255)),
            int(np.clip(round(g * dark_mult), 0, 255)),
            int(np.clip(round(b * dark_mult), 0, 255)),
        )
    else:
        # Feather slightly toward paper tone at dry-brush exit
        t = float(np.clip(shade_factor, 0.35, 1.0))
        return (
            int(np.clip(round(t * r + (1.0 - t) * pr), 0, 255)),
            int(np.clip(round(t * g + (1.0 - t) * pg), 0, 255)),
            int(np.clip(round(t * b + (1.0 - t) * pb), 0, 255)),
        )


def _jitter_rgb_hue(
    base_rgb: tuple[int, int, int],
    jitter_strength: float,
    seed: int,
) -> tuple[int, int, int]:
    """Shift a stroke's RGB hue and saturation deterministically for Impressionist broken-color shimmer."""
    if jitter_strength <= 0.01:
        return base_rgb
    rng = np.random.default_rng(seed)
    r, g, b = (c / 255.0 for c in base_rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Shift hue by up to +-0.35 turns at full jitter_strength
    dh = float(rng.uniform(-0.35, 0.35)) * jitter_strength
    h_new = (h + dh) % 1.0
    # Boost saturation slightly if source is near-monochrome so hue shimmer is visible
    s_min = min(0.55, 0.48 * jitter_strength)
    s_new = float(np.clip(max(s, s_min) + rng.uniform(-0.12, 0.18) * jitter_strength, 0.0, 1.0))
    l_new = float(np.clip(l + rng.uniform(-0.06, 0.06) * jitter_strength, 0.04, 0.92))
    rn, gn, bn = colorsys.hls_to_rgb(h_new, l_new, s_new)
    return (
        int(np.clip(round(rn * 255.0), 0, 255)),
        int(np.clip(round(gn * 255.0), 0, 255)),
        int(np.clip(round(bn * 255.0), 0, 255)),
    )


@dataclass
class BristleFilament:
    """Single dry-brush bristle sub-filament polygon along a stroke."""

    polygon_xy: np.ndarray               # Closed (M, 2) polygon coordinates in pixel space
    gray_level: float                    # Gray luminance in [0.0 (black), 1.0 (white)]
    rgb_color: tuple[int, int, int]      # Bristle RGB color
    opacity: float                       # Filament alpha in [0.0, 1.0]


@dataclass
class PainterlyStroke:
    """Variable-width brush stroke with color/tonal gradient & bristle filaments."""

    stroke_id: int
    layer_index: int
    spine_xy: np.ndarray                 # (N, 2) smoothed centerline in pixel space
    radii_px: np.ndarray                 # (N,) variable radius profile in pixels
    densities: np.ndarray                # (N,) sampled optical density D(t) in [0, 1]
    envelope_xy: np.ndarray              # (M, 2) closed polygon of main variable-width brush footprint
    # 3-stop tonal & RGB gradient along spine: (start, mid, end)
    gray_stops: tuple[float, float, float]
    rgb_stops: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    opacity_stops: tuple[float, float, float]
    bristles: list[BristleFilament] = field(default_factory=list)


@dataclass
class PainterlyLayer:
    """Single color/tonal slice layer stacked back-to-front."""

    layer_index: int                     # 0 = lightest wash, K-1 = darkest accent
    layer_name: str
    density_threshold: float
    target_gray: float                   # Characteristic gray luminance in [0.0, 1.0]
    target_rgb: tuple[int, int, int]     # Active RGB color for this layer (extracted or user-modified)
    color_hex: str                       # Active hex color (#rrggbb)
    auto_color_hex: str                  # Original hex color extracted from the source image
    base_opacity: float                  # Base layer opacity
    blur_sigma: float                    # Active Gaussian blur stdDeviation (px) for this layer
    default_blur_sigma: float            # Default Gaussian blur stdDeviation (px) for this layer
    mask: np.ndarray                     # Binary mask M_k for this layer
    edt_radius_px: np.ndarray            # Distance field R_k(x, y)
    skeleton: np.ndarray                 # Pruned skeleton K_k
    strokes: list[PainterlyStroke]


@dataclass
class PainterlyDocument:
    """Complete multi-layer painterly vector document."""

    width_px: int
    height_px: int
    paper_rgb: tuple[int, int, int]
    paper_hex: str
    luminance_raw: np.ndarray
    rgb_image: np.ndarray                # Source RGB image (H, W, 3) uint8
    density_map: np.ndarray
    bridged_density_map: np.ndarray
    layers: list[PainterlyLayer]

    @property
    def total_strokes(self) -> int:
        return sum(len(layer.strokes) for layer in self.layers)


def load_rgb_image(image_path: str | Path, paper_rgb: tuple[int, int, int] = (244, 241, 234)) -> np.ndarray:
    """Load source image as an 8-bit RGB array (H, W, 3), compositing RGBA over paper_rgb."""
    img = Image.open(image_path)
    if img.mode == "RGBA" or "transparency" in img.info:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float64)
        alpha = rgba[:, :, 3:4] / 255.0
        bg = np.array(paper_rgb, dtype=np.float64).reshape((1, 1, 3))
        comp = rgba[:, :, :3] * alpha + bg * (1.0 - alpha)
        return np.clip(np.round(comp), 0, 255).astype(np.uint8)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def extract_paper_color(
    image_path: str | Path, config: PainterlyConfig
) -> tuple[tuple[int, int, int], str]:
    """Sample the natural paper tone from the brightest 82nd-97th percentile of the source image."""
    if not config.match_source_paper_tone:
        rgb = _hex_to_rgb(config.fallback_paper_hex)
        return rgb, _rgb_to_hex(rgb)

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    p82 = np.percentile(lum, 82)
    p97 = np.percentile(lum, 97)
    mask = (lum >= p82) & (lum <= max(p97, p82 + 1.0))

    if np.any(mask):
        r = int(np.clip(np.median(arr[:, :, 0][mask]), 210, 253))
        g = int(np.clip(np.median(arr[:, :, 1][mask]), 208, 252))
        b = int(np.clip(np.median(arr[:, :, 2][mask]), 200, 250))
    else:
        r, g, b = 244, 241, 234

    return (r, g, b), _rgb_to_hex((r, g, b))


def compute_tonal_thresholds(density_map: np.ndarray, config: PainterlyConfig) -> list[float]:
    """Compute K monotonically increasing optical density thresholds [d_1, ..., d_K]."""
    if config.layer_thresholds is not None and len(config.layer_thresholds) >= 2:
        return sorted(float(x) for x in config.layer_thresholds)

    k = config.num_tonal_layers
    d_min = config.min_density_threshold
    d_max = config.max_density_threshold

    active = density_map[density_map >= d_min]
    uniform_schedule = np.linspace(d_min, d_max, k)

    if active.size > 100:
        quantiles = np.linspace(15.0, 90.0, k)
        q_vals = np.percentile(active, quantiles)
        q_vals = np.clip(q_vals, d_min, d_max)
        blended = 0.60 * uniform_schedule + 0.40 * q_vals
        for i in range(1, k):
            if blended[i] <= blended[i - 1] + 0.03:
                blended[i] = min(0.96, blended[i - 1] + 0.04)
        return [round(float(x), 4) for x in blended]

    return [round(float(x), 4) for x in uniform_schedule]


def extract_palette_and_layer_masks(
    rgb_image: np.ndarray,
    bridged_density: np.ndarray,
    thresholds: list[float],
    paper_rgb: tuple[int, int, int],
    config: PainterlyConfig,
) -> tuple[list[tuple[int, int, int]], list[np.ndarray]]:
    """Inspect the source image and restrict its colors to K layer palette entries and masks.

    In 'auto_palette' mode:
    - Runs deterministic K-Means clustering on foreground pixels in (R, G, B, Density) space
      to find K representative paint colors ordered from lightest wash to darkest accent.
    - Constructs each layer's binary stroke mask M_k combining color-cluster affinity and
      underpainting continuity so layers stack cleanly without gaps.
    """
    k = len(thresholds)
    h, w = bridged_density.shape
    res_scale = max(1.0, float(max(w, h)) / 480.0)
    close_k = max(3, int(round(3.0 * res_scale)) | 1)
    close_struct = np.ones((close_k, close_k), dtype=bool)
    d_min = thresholds[0]

    if config.color_palette_mode == "monochrome":
        palette_rgbs: list[tuple[int, int, int]] = []
        masks: list[np.ndarray] = []
        for idx, thr in enumerate(thresholds):
            layer_frac = idx / max(k - 1, 1)
            t_gray = float(np.clip(1.0 - (thr * 0.96 + 0.04 * layer_frac), 0.02, 0.88))
            rgb = (
                int(np.clip(round(t_gray * paper_rgb[0]), 0, 255)),
                int(np.clip(round(t_gray * paper_rgb[1]), 0, 255)),
                int(np.clip(round(t_gray * paper_rgb[2]), 0, 255)),
            )
            palette_rgbs.append(rgb)
            m = bridged_density > thr
            if np.any(m):
                m = ndimage.binary_closing(m, structure=close_struct)
            masks.append(m)
        return palette_rgbs, masks

    # Inspect foreground pixels where bridged_density > d_min
    fg_mask = bridged_density > d_min
    fg_ys, fg_xs = np.where(fg_mask)

    if len(fg_xs) < k * 10:
        # Fallback if image is almost blank
        palette_rgbs = []
        masks = []
        for idx, thr in enumerate(thresholds):
            g = int(np.clip(round((1.0 - thr) * 240), 10, 230))
            palette_rgbs.append((g, g, g))
            masks.append(bridged_density > thr)
        return palette_rgbs, masks

    # Subsample up to 12,000 foreground pixels deterministically for fast K-Means palette extraction
    rng = np.random.default_rng(42)
    if len(fg_xs) > 12000:
        sample_idx = rng.choice(len(fg_xs), size=12000, replace=False)
        s_ys = fg_ys[sample_idx]
        s_xs = fg_xs[sample_idx]
    else:
        s_ys = fg_ys
        s_xs = fg_xs

    fg_rgb = rgb_image[s_ys, s_xs].astype(np.float64)
    fg_d = bridged_density[s_ys, s_xs].astype(np.float64)

    # Check overall chromaticity (saturation) of the source artwork
    chroma = np.max(fg_rgb, axis=1) - np.min(fg_rgb, axis=1)
    mean_chroma = float(np.mean(chroma)) / 255.0

    # Feature vector combines RGB with weighted optical density so K-Means respects both hue and tone
    features = np.column_stack([fg_rgb, fg_d * 180.0])

    # Initialize centroids along the tonal threshold quantiles for deterministic convergence
    init_centroids = np.empty((k, 4), dtype=np.float64)
    for idx, thr in enumerate(thresholds):
        next_thr = thresholds[idx + 1] if idx + 1 < k else 1.01
        band = (fg_d >= thr) & (fg_d < next_thr)
        if not np.any(band):
            band = fg_d >= thr
        if np.any(band):
            init_centroids[idx, :3] = np.median(fg_rgb[band], axis=0)
            init_centroids[idx, 3] = float(np.median(fg_d[band])) * 180.0
        else:
            g_val = (1.0 - thr) * 255.0
            init_centroids[idx] = [g_val, g_val, g_val, thr * 180.0]

    try:
        centroids, _labels = cluster.vq.kmeans2(
            features,
            init_centroids,
            iter=18,
            minit="matrix",
        )
    except Exception:
        centroids = init_centroids

    # Sort centroids from lightest (highest luminance / lowest density) to darkest (highest density)
    centroid_lums = 0.2126 * centroids[:, 0] + 0.7152 * centroids[:, 1] + 0.0722 * centroids[:, 2]
    order = np.argsort(-centroid_lums)
    centroids = centroids[order]

    palette_rgbs = []
    if mean_chroma <= 0.08:
        # Low-chroma / monochrome source: compute clean tonal ramp tinted by the image's natural warmth
        mean_tint = np.mean(fg_rgb, axis=0)
        mean_lum = max(1.0, float(0.2126 * mean_tint[0] + 0.7152 * mean_tint[1] + 0.0722 * mean_tint[2]))
        tint_ratios = np.clip(mean_tint / mean_lum, 0.90, 1.10)
        for idx, thr in enumerate(thresholds):
            layer_frac = idx / max(k - 1, 1)
            t_gray = float(np.clip(1.0 - (thr * 0.92 + 0.06 * layer_frac), 0.03, 0.84))
            rgb = (
                int(np.clip(round(t_gray * paper_rgb[0] * tint_ratios[0]), 0, 232)),
                int(np.clip(round(t_gray * paper_rgb[1] * tint_ratios[1]), 0, 232)),
                int(np.clip(round(t_gray * paper_rgb[2] * tint_ratios[2]), 0, 232)),
            )
            palette_rgbs.append(rgb)
    else:
        for idx in range(k):
            r, g, b = (int(np.clip(round(centroids[idx, c]), 0, 255)) for c in range(3))
            # Prevent faintest wash centroid from collapsing to near-white paper halo
            lum_c = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if lum_c > 232.0:
                scale_c = 225.0 / lum_c
                r = int(np.clip(round(r * scale_c), 0, 255))
                g = int(np.clip(round(g * scale_c), 0, 255))
                b = int(np.clip(round(b * scale_c), 0, 255))
            palette_rgbs.append((r, g, b))

    # Build layer masks M_k:
    # If the source image has strong chromatic color (mean_chroma > 0.08), assign foreground pixels
    # by their color affinity to centroid k (plus darker compatible interior regions for underpainting).
    # If the source image is low-saturation / tonal brushwork, use progressive tonal slicing masks.
    masks = []
    if mean_chroma > 0.08:
        all_fg_rgb = rgb_image[fg_ys, fg_xs].astype(np.float64)
        all_fg_d = bridged_density[fg_ys, fg_xs].astype(np.float64)

        # Compute RGB distance from each foreground pixel to each of the K palette colors
        rgb_centroids = centroids[:, :3]
        dists = np.linalg.norm(
            all_fg_rgb[:, None, :] - rgb_centroids[None, :, :], axis=2
        )  # shape (N_fg, K)
        nearest_k = np.argmin(dists, axis=1)

        for idx, thr in enumerate(thresholds):
            c_rgb = rgb_centroids[idx]
            color_dist = dists[:, idx] / (255.0 * math.sqrt(3.0))

            # Include pixels where:
            # 1) Cluster idx is their nearest palette color, OR
            # 2) They are darker pixels (nearest_k >= idx and D >= thr) within close color affinity (< 0.22)
            #    so lighter color washes form a continuous underpainting beneath darker shading of the same hue
            active_fg = (nearest_k == idx) | (
                (nearest_k > idx) & (all_fg_d >= thr) & (color_dist < 0.22)
            )
            m = np.zeros((h, w), dtype=bool)
            m[fg_ys[active_fg], fg_xs[active_fg]] = True
            if np.any(m):
                m = ndimage.binary_closing(m, structure=close_struct)
            masks.append(m)
    else:
        for idx, thr in enumerate(thresholds):
            m = bridged_density > thr
            if np.any(m):
                m = ndimage.binary_closing(m, structure=close_struct)
            masks.append(m)

    return palette_rgbs, masks


def _enforce_forward_monotonic_side(
    side_pts: np.ndarray,
    spine_xy: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
) -> np.ndarray:
    """Prevent normal offset cusps/loops when stroke radius exceeds local radius of curvature."""
    out = side_pts.copy()
    n = len(out)
    if n < 2:
        return out

    # Fast vectorized check: if all consecutive steps project forward along tangent, no cusps exist
    dx = np.diff(out[:, 0])
    dy = np.diff(out[:, 1])
    proj = dx * tx[1:] + dy * ty[1:]
    if not np.any(proj < 0.0):
        return out

    first_bad = int(np.argmax(proj < 0.0)) + 1
    out_list = out.tolist()
    tx_list = tx.tolist()
    ty_list = ty.tolist()
    prev_x, prev_y = out_list[first_bad - 1]
    for i in range(first_bad, n):
        curr_x, curr_y = out_list[i]
        if (curr_x - prev_x) * tx_list[i] + (curr_y - prev_y) * ty_list[i] < 0.0:
            out_list[i][0] = prev_x
            out_list[i][1] = prev_y
        else:
            prev_x, prev_y = curr_x, curr_y
    return np.asarray(out_list, dtype=np.float64)


def _build_rounded_variable_width_polygon(
    spine_xy: np.ndarray,
    radii: np.ndarray,
    edge_roughness: float = 0.0,
    seed: int = 0,
    nib_angle_deg: float = 35.0,
    nib_flatness: float = 0.0,
    flat_caps: bool = False,
) -> np.ndarray:
    """Construct a cusp-free closed 2D polygon around spine_xy with half-width radii and rounded/flat caps."""
    n = len(spine_xy)
    if n < 2:
        return spine_xy.copy()

    max_r = float(np.max(radii))
    tangent_sigma = min(max(1.5, 0.35 * max_r), max(1.0, n / 3.0))
    smooth_x = ndimage.gaussian_filter1d(spine_xy[:, 0], sigma=tangent_sigma)
    smooth_y = ndimage.gaussian_filter1d(spine_xy[:, 1], sigma=tangent_sigma)

    dx = np.gradient(smooth_x)
    dy = np.gradient(smooth_y)
    norms = np.hypot(dx, dy)
    norms = np.where(norms > 1e-8, norms, 1.0)
    tx = dx / norms
    ty = dy / norms
    nx_vec = -ty
    ny_vec = tx

    eff_radii = radii
    if nib_flatness > 0.01:
        nib_rad = math.radians(nib_angle_deg)
        nib_dx = math.cos(nib_rad)
        nib_dy = math.sin(nib_rad)
        # Cross product magnitude between stroke tangent and chisel nib edge
        cross = np.abs(tx * nib_dy - ty * nib_dx)
        nib_scale = (1.0 - nib_flatness) + nib_flatness * np.clip(cross, 0.06, 1.0)
        eff_radii = np.maximum(0.30, radii * nib_scale)

    xs = spine_xy[:, 0]
    ys = spine_xy[:, 1]

    if edge_roughness > 0.0 and n >= 6:
        rng = np.random.default_rng(seed)
        noise_l = ndimage.gaussian_filter1d(rng.normal(0.0, 1.0, size=n), sigma=2.5)
        noise_r = ndimage.gaussian_filter1d(rng.normal(0.0, 1.0, size=n), sigma=2.5)
        r_left = np.maximum(0.35, eff_radii * (1.0 + edge_roughness * 0.14 * noise_l))
        r_right = np.maximum(0.35, eff_radii * (1.0 + edge_roughness * 0.14 * noise_r))
    else:
        r_left = eff_radii
        r_right = eff_radii

    left_raw = np.column_stack([xs + r_left * nx_vec, ys + r_left * ny_vec])
    right_raw = np.column_stack([xs - r_right * nx_vec, ys - r_right * ny_vec])

    left_pts = _enforce_forward_monotonic_side(left_raw, spine_xy, tx, ty)
    right_pts = _enforce_forward_monotonic_side(right_raw, spine_xy, tx, ty)[::-1]

    r0 = float(eff_radii[0])
    start_cap: list[np.ndarray] = []
    if r0 >= 1.2 and not flat_caps:
        base_angle = math.atan2(float(ny_vec[0]), float(nx_vec[0]))
        angles = np.linspace(base_angle + 0.5 * math.pi, base_angle + 1.5 * math.pi, 7)[1:-1]
        cap_x = xs[0] + r0 * np.cos(angles)
        cap_y = ys[0] + r0 * np.sin(angles)
        start_cap = [np.column_stack([cap_x, cap_y])]

    r_end = float(eff_radii[-1])
    end_cap: list[np.ndarray] = []
    if r_end >= 1.2 and not flat_caps:
        base_angle = math.atan2(float(ny_vec[-1]), float(nx_vec[-1]))
        angles = np.linspace(base_angle + 0.5 * math.pi, base_angle - 0.5 * math.pi, 7)[1:-1]
        cap_x = xs[-1] + r_end * np.cos(angles)
        cap_y = ys[-1] + r_end * np.sin(angles)
        end_cap = [np.column_stack([cap_x, cap_y])]

    parts = [left_pts] + end_cap + [right_pts] + start_cap
    return np.vstack(parts)


def _generate_bristle_filaments(
    spine_xy: np.ndarray,
    radii_px: np.ndarray,
    densities: np.ndarray,
    base_gray: float,
    base_rgb: tuple[int, int, int],
    paper_rgb: tuple[int, int, int],
    base_opacity: float,
    config: PainterlyConfig,
    seed: int,
) -> list[BristleFilament]:
    """Generate parallel dry-brush bristle sub-filaments that create 'flying white' texture."""
    if config.bristle_count <= 0 or (config.dry_brush_strength <= 0.02 and config.bristle_splay <= 0.20):
        return []

    max_r = float(np.max(radii_px))
    n = len(spine_xy)
    if max_r < 2.2 or n < 4:
        return []

    rng = np.random.default_rng(seed)
    tangent_sigma = min(max(1.5, 0.35 * max_r), max(1.0, n / 3.0))
    smooth_x = ndimage.gaussian_filter1d(spine_xy[:, 0], sigma=tangent_sigma)
    smooth_y = ndimage.gaussian_filter1d(spine_xy[:, 1], sigma=tangent_sigma)
    dx = np.gradient(smooth_x)
    dy = np.gradient(smooth_y)
    norms = np.maximum(np.hypot(dx, dy), 1e-8)
    nx_vec = -dy / norms
    ny_vec = dx / norms

    num_bristles = min(config.bristle_count, max(2, int(round(max_r / 2.2))))
    frac_offsets = np.linspace(-0.75, 0.75, num_bristles)
    filaments: list[BristleFilament] = []

    eff_dry = max(config.dry_brush_strength, min(1.0, config.bristle_splay * 0.45))

    for b_idx, frac in enumerate(frac_offsets):
        start_idx = 0 if (b_idx % 2 == 0) else int(0.18 * n)
        end_idx = n
        if end_idx - start_idx < 3:
            continue

        sub_s = spine_xy[start_idx:end_idx]
        sub_r = radii_px[start_idx:end_idx]
        sub_nx = nx_vec[start_idx:end_idx]
        sub_ny = ny_vec[start_idx:end_idx]
        m = len(sub_s)

        t_local = np.linspace(0.0, 1.0, m)
        splay = 1.0 + config.bristle_splay * eff_dry * (t_local ** 1.4)
        wave_amp = 0.06 * (1.0 + 0.35 * max(0.0, config.bristle_splay - 0.14))
        wave = ndimage.gaussian_filter1d(rng.normal(0.0, wave_amp, size=m), sigma=max(1.0, min(2.2, m / 3.0)))
        offset_dist = (frac * splay + wave) * sub_r

        b_spine = np.column_stack([
            sub_s[:, 0] + offset_dist * sub_nx,
            sub_s[:, 1] + offset_dist * sub_ny,
        ])

        filament_r = np.maximum(
            0.35,
            (sub_r / (num_bristles + 1.6)) * (1.0 - 0.45 * (t_local ** 1.6)),
        )
        poly = _build_rounded_variable_width_polygon(
            b_spine,
            filament_r,
            edge_roughness=0.0,
            seed=seed + b_idx + 1,
            flat_caps=config.polygon_decimation > 0.35,
        )

        shade_mult = float(rng.uniform(1.06, 1.25))
        b_rgb = _modulate_rgb_tone(base_rgb, paper_rgb, shade_mult)
        b_gray = float(np.clip(base_gray * rng.uniform(0.65, 0.92), 0.01, 0.92))
        b_opacity = float(
            np.clip(base_opacity * eff_dry * rng.uniform(0.42, 0.72), 0.10, 0.88)
        )
        filaments.append(
            BristleFilament(
                polygon_xy=poly,
                gray_level=b_gray,
                rgb_color=b_rgb,
                opacity=b_opacity,
            )
        )

    return filaments


def _parameterize_painterly_stroke(
    path: PixelPath,
    stroke_id: int,
    layer_index: int,
    num_layers: int,
    layer_max_radius_px: float,
    target_gray: float,
    target_rgb: tuple[int, int, int],
    paper_rgb: tuple[int, int, int],
    has_custom_color: bool,
    base_opacity: float,
    config: PainterlyConfig,
    res_scale: float = 1.0,
) -> PainterlyStroke | None:
    """Fit a 3D B-spline (x(t), y(t), r(t)) and build painterly stroke geometry with RGB stops."""
    coords = path.coords_xy_px
    radii = np.clip(path.radii_px * config.stroke_width_scale, 0.6 * res_scale, layer_max_radius_px * 1.25)
    densities = np.clip(path.densities, 0.05, 1.0)

    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    keep = np.concatenate([[True], seg_lens > 1e-4])
    coords = coords[keep]
    radii = radii[keep]
    densities = densities[keep]

    n_pts = len(coords)
    if n_pts < 2:
        return None

    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum_len[-1])
    if total_len < config.min_stroke_length_px * res_scale:
        return None

    u_chord = cum_len / max(total_len, 1e-4)
    step_px = max(1.0, 2.0 * res_scale)
    base_samples = max(14, min(140, int(math.ceil(total_len / step_px)) + 1))
    if config.polygon_decimation > 0.01:
        # Reduce sample count down to 3..14 for low-poly origami / faceted crystal strokes
        max_decimated = max(3, int(round(14.0 * (1.0 - 0.78 * config.polygon_decimation))))
        num_samples = max(3, min(base_samples, max_decimated))
    else:
        num_samples = base_samples
    t_vals = np.linspace(0.0, 1.0, num_samples)

    smooth_r = ndimage.gaussian_filter1d(radii, sigma=min(2.5 * res_scale, max(0.5, n_pts / 4.0)))
    degree = min(3, n_pts - 1)
    s_factor = config.spline_smoothing_factor * n_pts * (res_scale ** 2)

    try:
        tck, _u = interpolate.splprep(
            [coords[:, 0], coords[:, 1], smooth_r],
            u=u_chord,
            k=degree,
            s=s_factor if n_pts > 4 else 0.0,
        )
        x_t, y_t, r_t = interpolate.splev(t_vals, tck, der=0)
        x_t = np.asarray(x_t, dtype=np.float64)
        y_t = np.asarray(y_t, dtype=np.float64)
        r_t = np.asarray(r_t, dtype=np.float64)

        margin = 3.5 * res_scale
        if (
            np.min(x_t) < np.min(coords[:, 0]) - margin
            or np.max(x_t) > np.max(coords[:, 0]) + margin
            or np.min(y_t) < np.min(coords[:, 1]) - margin
            or np.max(y_t) > np.max(coords[:, 1]) + margin
        ):
            raise ValueError("Overshoot")
    except Exception:
        x_t = np.interp(t_vals, u_chord, coords[:, 0])
        y_t = np.interp(t_vals, u_chord, coords[:, 1])
        r_t = np.interp(t_vals, u_chord, smooth_r)

    # Optional Tangent Over-Extension (sketchy architectural overshoots)
    if config.stroke_extension_px > 0.05 and len(x_t) >= 2:
        ext = float(config.stroke_extension_px) * res_scale
        k_look = min(3, len(x_t) - 1)
        dx0 = x_t[k_look] - x_t[0]
        dy0 = y_t[k_look] - y_t[0]
        n0 = max(math.hypot(dx0, dy0), 1e-6)
        tx0, ty0 = dx0 / n0, dy0 / n0

        dx1 = x_t[-1] - x_t[-1 - k_look]
        dy1 = y_t[-1] - y_t[-1 - k_look]
        n1 = max(math.hypot(dx1, dy1), 1e-6)
        tx1, ty1 = dx1 / n1, dy1 / n1

        # Smoothly push start backwards along -t0 and end forwards along +t1
        w_start = (1.0 - t_vals) ** 2
        w_end = t_vals ** 2
        x_t = x_t - ext * w_start * tx0 + ext * w_end * tx1
        y_t = y_t - ext * w_start * ty0 + ext * w_end * ty1

    # Optional Spine Wave / Tremor Oscillation (Van Gogh swirls or woodcut tremors)
    if config.spine_wave_amplitude > 0.05 and len(x_t) >= 3:
        dx_w = np.gradient(x_t)
        dy_w = np.gradient(y_t)
        norm_w = np.maximum(np.hypot(dx_w, dy_w), 1e-6)
        nx_w = -dy_w / norm_w
        ny_w = dx_w / norm_w
        phase = (stroke_id * 1.6180339887 + layer_index * 0.75) % (2.0 * math.pi)
        wave_disp = (config.spine_wave_amplitude * res_scale) * np.sin(
            2.0 * math.pi * config.spine_wave_frequency * t_vals + phase
        )
        x_t = x_t + wave_disp * nx_w
        y_t = y_t + wave_disp * ny_w

    r_t = np.clip(r_t, 0.5 * res_scale, layer_max_radius_px * 1.25)

    # Optional Radius Power-Law Gamma (1 = natural, >1 = exaggerated contrast, 0 = monoline, <0 = inverted)
    if abs(config.radius_gamma - 1.0) > 1e-3:
        ref_r = max(layer_max_radius_px * 0.45, 1.5 * res_scale)
        r_ratio = np.clip(r_t / ref_r, 0.08, 3.0)
        r_t = ref_r * (r_ratio ** config.radius_gamma) * config.stroke_width_scale
        r_t = np.clip(r_t, 0.35 * res_scale, layer_max_radius_px * 2.4)

    d_t = np.clip(np.interp(t_vals, u_chord, densities), 0.05, 1.0)

    # Taper profile: 0.0 = blunt tape, 1.0 = natural brush, >1.0 = sharp comet whip, <0.0 = inverted teardrop
    tp = config.taper_power
    taper_window = np.ones_like(t_vals)
    if abs(tp - 1.0) <= 1e-3:
        entry_zone = t_vals < 0.08
        exit_zone = t_vals > 0.86
        taper_window[entry_zone] = 0.65 + 0.35 * (t_vals[entry_zone] / 0.08)
        taper_window[exit_zone] = 1.0 - 0.50 * ((t_vals[exit_zone] - 0.86) / 0.14)
    elif tp > 0.01:
        if tp < 1.0:
            # Interpolate between blunt (tp=0) and natural (tp=1)
            nat = np.ones_like(t_vals)
            entry_zone = t_vals < 0.08
            exit_zone = t_vals > 0.86
            nat[entry_zone] = 0.65 + 0.35 * (t_vals[entry_zone] / 0.08)
            nat[exit_zone] = 1.0 - 0.50 * ((t_vals[exit_zone] - 0.86) / 0.14)
            taper_window = (1.0 - tp) * 1.0 + tp * nat
        else:
            # Comet tail / whip lash taper along the stroke
            whip_exp = 0.55 * (tp - 0.5)
            entry_ramp = np.minimum(1.0, 0.55 + 0.45 * (t_vals / 0.08))
            exit_whip = np.maximum(0.08, (1.0 - 0.88 * (t_vals ** 1.2)) ** whip_exp)
            taper_window = entry_ramp * exit_whip
    elif tp < -0.01:
        # Inverted teardrop / tadpole (needle thin at start, bulbous head at end)
        inv_exp = 0.65 * abs(tp)
        taper_window = np.maximum(0.10, (0.12 + 0.88 * t_vals) ** inv_exp)

    r_t = np.maximum(0.35, r_t * taper_window)

    use_flat_caps = (abs(tp) < 0.12) or (config.polygon_decimation > 0.35)
    spine_xy = np.column_stack([x_t, y_t])
    envelope_xy = _build_rounded_variable_width_polygon(
        spine_xy,
        r_t,
        edge_roughness=config.dry_brush_strength * 0.45 * (1.0 - 0.8 * config.polygon_decimation),
        seed=stroke_id * 31 + layer_index,
        nib_angle_deg=config.nib_angle_deg,
        nib_flatness=config.nib_flatness,
        flat_caps=use_flat_caps,
    )

    # Optional Per-Stroke Hue Jitter (Impressionist broken-color shimmer)
    stroke_base_rgb = (
        _jitter_rgb_hue(target_rgb, config.stroke_hue_jitter, stroke_id * 131 + layer_index * 17)
        if config.stroke_hue_jitter > 0.01
        else target_rgb
    )

    # Compute 3-stop tonal & RGB gradient along the stroke spine
    d_start = float(np.mean(d_t[: max(2, num_samples // 4)]))
    d_mid = float(np.mean(d_t[num_samples // 4 : max(3, (3 * num_samples) // 4)]))
    d_end = float(np.mean(d_t[max(0, (3 * num_samples) // 4) :]))

    g_start = float(np.clip(0.30 * target_gray + 0.70 * (1.0 - d_start ** 1.1), 0.01, 0.94))
    g_mid = float(np.clip(0.30 * target_gray + 0.70 * (1.0 - d_mid ** 1.1), 0.01, 0.94))
    g_end = float(np.clip(0.30 * target_gray + 0.70 * (1.0 - d_end ** 1.1) + 0.03, 0.01, 0.96))

    # Modulate the stroke's base RGB across start (heavy press), mid, and end (dry lift)
    mean_d = max(float(np.mean(d_t)), 0.1)
    if has_custom_color or config.stroke_hue_jitter > 0.01:
        # Preserve the chosen/jittered hue while adding natural press-to-lift shading
        s_factor_start = float(np.clip(1.06 * (d_start / mean_d), 0.82, 1.22))
        s_factor_mid = 1.0
        s_factor_end = float(np.clip(0.90 * (d_end / mean_d), 0.75, 1.08))
    else:
        # Combine the layer's extracted RGB color with local stroke density shading
        layer_d_ref = max(1.0 - target_gray, 0.12)
        s_factor_start = float(np.clip((d_start / layer_d_ref) ** 0.45, 0.72, 1.35))
        s_factor_mid = float(np.clip((d_mid / layer_d_ref) ** 0.45, 0.72, 1.35))
        s_factor_end = float(np.clip((d_end / layer_d_ref) ** 0.45 * 0.92, 0.68, 1.25))

    rgb_start = _modulate_rgb_tone(stroke_base_rgb, paper_rgb, s_factor_start)
    rgb_mid = _modulate_rgb_tone(stroke_base_rgb, paper_rgb, s_factor_mid)
    rgb_end = _modulate_rgb_tone(stroke_base_rgb, paper_rgb, s_factor_end)

    op_start = float(np.clip(base_opacity * (0.85 + 0.25 * d_start), 0.12, 0.96))
    op_mid = float(np.clip(base_opacity * (0.82 + 0.25 * d_mid), 0.12, 0.95))
    op_end = float(np.clip(base_opacity * (0.60 + 0.25 * d_end), 0.08, 0.90))

    bristles: list[BristleFilament] = []
    if layer_index >= 1:
        bristles = _generate_bristle_filaments(
            spine_xy=spine_xy,
            radii_px=r_t,
            densities=d_t,
            base_gray=g_mid,
            base_rgb=rgb_mid,
            paper_rgb=paper_rgb,
            base_opacity=op_mid,
            config=config,
            seed=stroke_id * 97 + layer_index,
        )

    return PainterlyStroke(
        stroke_id=stroke_id,
        layer_index=layer_index,
        spine_xy=spine_xy,
        radii_px=r_t,
        densities=d_t,
        envelope_xy=envelope_xy,
        gray_stops=(round(g_start, 4), round(g_mid, 4), round(g_end, 4)),
        rgb_stops=(rgb_start, rgb_mid, rgb_end),
        opacity_stops=(round(op_start, 4), round(op_mid, 4), round(op_end, 4)),
        bristles=bristles,
    )


# Bounded LRU cache for Stage 1-5 topology results so brush appearance sliders
# (width scale, bristles, dry-brush, spline smoothing, opacity, colors, blur) run in ~15ms
_TOPOLOGY_CACHE: dict[tuple, dict] = {}
_MAX_TOPOLOGY_CACHE = 4


def run_painterly_pipeline(
    image_path: str | Path,
    config: PainterlyConfig | None = None,
) -> PainterlyDocument:
    """Execute the multi-layer color & tonal painterly pipeline."""
    cfg = config or PainterlyConfig()
    img_path = Path(image_path).resolve()
    try:
        mtime_ns = img_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0

    paper_rgb, paper_hex = extract_paper_color(img_path, cfg)

    topo_key = (
        str(img_path),
        mtime_ns,
        cfg.num_tonal_layers,
        cfg.color_palette_mode,
        tuple(cfg.layer_thresholds) if cfg.layer_thresholds else None,
        round(cfg.min_density_threshold, 4),
        round(cfg.max_density_threshold, 4),
        round(cfg.anisotropic_sigma, 4),
        round(cfg.flow_lock_strength, 4),
        round(cfg.flow_lock_angle_deg, 2),
        round(cfg.spur_prune_factor, 4),
        round(cfg.min_stroke_length_px, 4),
        round(cfg.max_deflection_deg, 4),
        cfg.normalize_paper_illumination,
    )

    cached = _TOPOLOGY_CACHE.get(topo_key)
    if cached is not None:
        lum_raw = cached["lum_raw"]
        rgb_image = cached["rgb_image"]
        density_map = cached["density_map"]
        bridged_density = cached["bridged_density"]
        thresholds = cached["thresholds"]
        auto_palette_rgbs = cached["auto_palette_rgbs"]
        layer_masks = cached["layer_masks"]
        per_layer_topo = cached["per_layer_topo"]
        height_px, width_px = lum_raw.shape[:2]
        max_dim = float(max(width_px, height_px))
        res_scale = max(1.0, max_dim / 480.0)
        num_layers = len(thresholds)
    else:
        lum_raw = load_luminance_image(img_path)
        height_px, width_px = lum_raw.shape[:2]
        max_dim = float(max(width_px, height_px))
        res_scale = max(1.0, max_dim / 480.0)
        rgb_image = load_rgb_image(img_path, paper_rgb=paper_rgb)

        if cfg.normalize_paper_illumination:
            lum_norm = normalize_paper_luminance(lum_raw)
        else:
            lum_norm = np.clip(lum_raw, 0.0, 1.0)

        density_map = 1.0 - lum_norm
        bridged_density = anisotropic_gap_bridging(
            density_map,
            sigma=cfg.anisotropic_sigma * res_scale,
            flow_lock_strength=cfg.flow_lock_strength,
            flow_lock_angle_deg=cfg.flow_lock_angle_deg,
        )

        thresholds = compute_tonal_thresholds(bridged_density, cfg)
        num_layers = len(thresholds)

        auto_palette_rgbs, layer_masks = extract_palette_and_layer_masks(
            rgb_image=rgb_image,
            bridged_density=bridged_density,
            thresholds=thresholds,
            paper_rgb=paper_rgb,
            config=cfg,
        )

        eff_min_len_px = cfg.min_stroke_length_px * res_scale
        stage_cfg = PhysicalCalibrationConfig(
            spur_prune_factor=cfg.spur_prune_factor,
            min_stroke_length_px=eff_min_len_px,
            max_deflection_deg=cfg.max_deflection_deg,
        )

        per_layer_topo = []
        for idx, _thr in enumerate(thresholds):
            layer_frac = idx / max(num_layers - 1, 1)
            layer_max_r_px = max(5.0 * res_scale, max_dim * (0.022 * (1.0 - 0.62 * layer_frac)))
            stepover_px = max(3.0 * res_scale, 0.62 * layer_max_r_px)
            mask = layer_masks[idx]

            # Stage 2: EDT + Skeletonization + Spur Pruning
            s2 = run_stage2_medial_axis(mask, stage_cfg)

            # Stage 3: Graph Instantiation + Bridge Collapse + Momentum Untangling
            s3 = run_stage3_graph_untangling(
                pruned_skeleton=s2.pruned_skeleton,
                edt_radius_px=s2.edt_radius_px,
                density_map=density_map,
                config=stage_cfg,
            )

            # Subdivide wide regions (> layer_max_r_px) into parallel flowing brushstrokes
            expanded_paths: list[PixelPath] = []
            for p in s3.untangled_paths:
                if float(np.max(p.radii_px)) > layer_max_r_px * 1.15:
                    subs = _subdivide_broad_wash(
                        path=p,
                        stroke_mask=mask,
                        edt_radius_px=s2.edt_radius_px,
                        density_map=density_map,
                        stepover_px=stepover_px,
                        r_max_px=layer_max_r_px,
                        min_length_px=eff_min_len_px,
                        start_id=len(expanded_paths),
                    )
                    if subs:
                        expanded_paths.extend(subs)
                    else:
                        expanded_paths.append(p)
                else:
                    expanded_paths.append(p)

            # Stage 5: Polarity Resolution (press -> lift)
            s5 = run_stage5_polarity_resolution(
                classified_paths=expanded_paths,
                density_map=density_map,
                config=stage_cfg,
            )
            sorted_paths = sorted(
                s5.oriented_paths,
                key=lambda p: float(np.max(p.radii_px)),
                reverse=True,
            )
            per_layer_topo.append(
                {
                    "edt_radius_px": s2.edt_radius_px,
                    "pruned_skeleton": s2.pruned_skeleton,
                    "untangled_paths": s3.untangled_paths,
                    "sorted_paths": sorted_paths,
                }
            )

        if len(_TOPOLOGY_CACHE) >= _MAX_TOPOLOGY_CACHE:
            oldest_key = next(iter(_TOPOLOGY_CACHE))
            _TOPOLOGY_CACHE.pop(oldest_key, None)
        _TOPOLOGY_CACHE[topo_key] = {
            "lum_raw": lum_raw,
            "rgb_image": rgb_image,
            "density_map": density_map,
            "bridged_density": bridged_density,
            "thresholds": thresholds,
            "auto_palette_rgbs": auto_palette_rgbs,
            "layer_masks": layer_masks,
            "per_layer_topo": per_layer_topo,
        }

    op_min, op_max = cfg.layer_opacity_range
    layer_names = [
        "faint_wash",
        "light_wash",
        "soft_midtone",
        "mid_charcoal",
        "dark_wash",
        "deep_ink",
        "black_accents",
    ]

    layers: list[PainterlyLayer] = []
    global_stroke_id = 0

    for idx, thr in enumerate(thresholds):
        layer_frac = idx / max(num_layers - 1, 1)
        l_name = (
            layer_names[idx]
            if idx < len(layer_names)
            else f"color_layer_{idx + 1}"
        )

        layer_max_r_px = max(5.0 * res_scale, max_dim * (0.022 * (1.0 - 0.62 * layer_frac)))
        mask = layer_masks[idx]
        auto_rgb = auto_palette_rgbs[idx]
        auto_hex = _rgb_to_hex(auto_rgb)

        has_custom = (
            cfg.layer_colors is not None
            and idx < len(cfg.layer_colors)
            and bool(cfg.layer_colors[idx])
        )
        if has_custom and cfg.layer_colors is not None and cfg.layer_colors[idx] is not None:
            active_rgb = _hex_to_rgb(str(cfg.layer_colors[idx]))
            active_hex = _rgb_to_hex(active_rgb)
        else:
            active_rgb = auto_rgb
            active_hex = auto_hex

        topo_item = per_layer_topo[idx]
        sorted_paths = topo_item["sorted_paths"]

        target_gray = float(
            (0.2126 * active_rgb[0] + 0.7152 * active_rgb[1] + 0.0722 * active_rgb[2]) / 255.0
        )
        base_opacity = float(op_min + (op_max - op_min) * (layer_frac ** 0.80))

        if idx < max(1, num_layers // 3):
            default_blur = 2.0
        elif idx < max(2, (2 * num_layers) // 3):
            default_blur = 0.66
        else:
            default_blur = 0.0

        if (
            cfg.layer_blurs is not None
            and idx < len(cfg.layer_blurs)
            and cfg.layer_blurs[idx] is not None
        ):
            active_blur = float(np.clip(float(cfg.layer_blurs[idx]), 0.0, 16.0))
        else:
            active_blur = default_blur

        layer_strokes: list[PainterlyStroke] = []
        for path in sorted_paths:
            p_stroke = _parameterize_painterly_stroke(
                path=path,
                stroke_id=global_stroke_id,
                layer_index=idx,
                num_layers=num_layers,
                layer_max_radius_px=layer_max_r_px,
                target_gray=target_gray,
                target_rgb=active_rgb,
                paper_rgb=paper_rgb,
                has_custom_color=has_custom,
                base_opacity=base_opacity,
                config=cfg,
                res_scale=res_scale,
            )
            if p_stroke is not None:
                layer_strokes.append(p_stroke)
                global_stroke_id += 1

        layers.append(
            PainterlyLayer(
                layer_index=idx,
                layer_name=f"{idx + 1}_{l_name}",
                density_threshold=thr,
                target_gray=target_gray,
                target_rgb=active_rgb,
                color_hex=active_hex,
                auto_color_hex=auto_hex,
                base_opacity=base_opacity,
                blur_sigma=active_blur,
                default_blur_sigma=default_blur,
                mask=mask,
                edt_radius_px=topo_item["edt_radius_px"],
                skeleton=topo_item["pruned_skeleton"],
                strokes=layer_strokes,
            )
        )

    return PainterlyDocument(
        width_px=width_px,
        height_px=height_px,
        paper_rgb=paper_rgb,
        paper_hex=paper_hex,
        luminance_raw=lum_raw,
        rgb_image=rgb_image,
        density_map=density_map,
        bridged_density_map=bridged_density,
        layers=layers,
    )


def _parameterize_plotter_layer_stroke(
    path: PixelPath,
    stroke_id: int,
    layer_index: int,
    layer_name: str,
    layer_max_radius_px: float,
    target_rgb: tuple[int, int, int],
    transform: CanvasTransform,
    p_cfg: PainterlyConfig,
    c_cfg: PhysicalCalibrationConfig,
    res_scale: float = 1.0,
) -> ParameterizedStrokeData | None:
    """Fit 3D B-spline centerline with Painterly mutations and Stage 6 physical kinematics."""
    coords = path.coords_xy_px
    radii = np.clip(path.radii_px * p_cfg.stroke_width_scale, 0.4 * res_scale, layer_max_radius_px * 1.25)
    densities = np.clip(path.densities, 0.05, 1.0)

    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    keep = np.concatenate([[True], seg_lens > 1e-4])
    coords = coords[keep]
    radii = radii[keep]
    densities = densities[keep]

    n_pts = len(coords)
    if n_pts < 2:
        return None

    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len_px = float(cum_len[-1])
    if total_len_px < p_cfg.min_stroke_length_px * res_scale:
        return None

    total_len_mm = max(total_len_px * transform.scale_mm_per_px, 1e-4)
    u_chord = cum_len / max(total_len_px, 1e-4)

    base_samples = max(
        c_cfg.min_waypoints,
        min(140, int(math.ceil(total_len_mm / c_cfg.waypoint_spacing_mm)) + 1),
    )
    if p_cfg.polygon_decimation > 0.01:
        max_decimated = max(3, int(round(14.0 * (1.0 - 0.78 * p_cfg.polygon_decimation))))
        num_samples = max(3, min(base_samples, max_decimated))
    else:
        num_samples = base_samples
    t_vals = np.linspace(0.0, 1.0, num_samples)

    smooth_r = ndimage.gaussian_filter1d(radii, sigma=min(2.5 * res_scale, max(0.5, n_pts / 4.0)))
    degree = min(3, n_pts - 1)
    s_factor = p_cfg.spline_smoothing_factor * n_pts * (res_scale ** 2)

    try:
        tck, _u = interpolate.splprep(
            [coords[:, 0], coords[:, 1], smooth_r],
            u=u_chord,
            k=degree,
            s=s_factor if n_pts > 4 else 0.0,
        )
        x_t, y_t, r_t = interpolate.splev(t_vals, tck, der=0)
        x_t = np.asarray(x_t, dtype=np.float64)
        y_t = np.asarray(y_t, dtype=np.float64)
        r_t = np.asarray(r_t, dtype=np.float64)

        margin = 3.5 * res_scale
        if (
            np.min(x_t) < np.min(coords[:, 0]) - margin
            or np.max(x_t) > np.max(coords[:, 0]) + margin
            or np.min(y_t) < np.min(coords[:, 1]) - margin
            or np.max(y_t) > np.max(coords[:, 1]) + margin
        ):
            raise ValueError("Overshoot")
    except Exception:
        x_t = np.interp(t_vals, u_chord, coords[:, 0])
        y_t = np.interp(t_vals, u_chord, coords[:, 1])
        r_t = np.interp(t_vals, u_chord, smooth_r)

    # Optional Tangent Over-Extension (sketchy architectural overshoots)
    if p_cfg.stroke_extension_px > 0.05 and len(x_t) >= 2:
        ext = float(p_cfg.stroke_extension_px) * res_scale
        k_look = min(3, len(x_t) - 1)
        dx0 = x_t[k_look] - x_t[0]
        dy0 = y_t[k_look] - y_t[0]
        n0 = max(math.hypot(dx0, dy0), 1e-6)
        tx0, ty0 = dx0 / n0, dy0 / n0

        dx1 = x_t[-1] - x_t[-1 - k_look]
        dy1 = y_t[-1] - y_t[-1 - k_look]
        n1 = max(math.hypot(dx1, dy1), 1e-6)
        tx1, ty1 = dx1 / n1, dy1 / n1

        w_start = (1.0 - t_vals) ** 2
        w_end = t_vals ** 2
        x_t = x_t - ext * w_start * tx0 + ext * w_end * tx1
        y_t = y_t - ext * w_start * ty0 + ext * w_end * ty1

    # Optional Spine Wave / Tremor Oscillation
    if p_cfg.spine_wave_amplitude > 0.05 and len(x_t) >= 3:
        dx_w = np.gradient(x_t)
        dy_w = np.gradient(y_t)
        norm_w = np.maximum(np.hypot(dx_w, dy_w), 1e-6)
        nx_w = -dy_w / norm_w
        ny_w = dx_w / norm_w
        phase = (stroke_id * 1.6180339887 + layer_index * 0.75) % (2.0 * math.pi)
        wave_disp = (p_cfg.spine_wave_amplitude * res_scale) * np.sin(
            2.0 * math.pi * p_cfg.spine_wave_frequency * t_vals + phase
        )
        x_t = x_t + wave_disp * nx_w
        y_t = y_t + wave_disp * ny_w

    r_t = np.clip(r_t, 0.35 * res_scale, layer_max_radius_px * 1.25)

    # Optional Radius Power-Law Gamma
    if abs(p_cfg.radius_gamma - 1.0) > 1e-3:
        ref_r = max(layer_max_radius_px * 0.45, 1.5 * res_scale)
        r_ratio = np.clip(r_t / ref_r, 0.08, 3.0)
        r_t = ref_r * (r_ratio ** p_cfg.radius_gamma) * p_cfg.stroke_width_scale
        r_t = np.clip(r_t, 0.30 * res_scale, layer_max_radius_px * 2.4)

    # Optional Calligraphic Chisel Nib Flatness modulation along tangent angle
    if p_cfg.nib_flatness > 0.01 and len(x_t) >= 2:
        dx_nib = np.gradient(x_t)
        dy_nib = np.gradient(y_t)
        norm_nib = np.maximum(np.hypot(dx_nib, dy_nib), 1e-6)
        tx_nib = dx_nib / norm_nib
        ty_nib = dy_nib / norm_nib
        nib_rad = math.radians(p_cfg.nib_angle_deg)
        cross = np.abs(tx_nib * math.sin(nib_rad) - ty_nib * math.cos(nib_rad))
        nib_scale = (1.0 - p_cfg.nib_flatness) + p_cfg.nib_flatness * np.clip(cross, 0.06, 1.0)
        r_t = np.maximum(0.30, r_t * nib_scale)

    # Taper profile along stroke spine
    tp = p_cfg.taper_power
    taper_window = np.ones_like(t_vals)
    if abs(tp - 1.0) <= 1e-3:
        entry_zone = t_vals < 0.08
        exit_zone = t_vals > 0.86
        taper_window[entry_zone] = 0.65 + 0.35 * (t_vals[entry_zone] / 0.08)
        taper_window[exit_zone] = 1.0 - 0.50 * ((t_vals[exit_zone] - 0.86) / 0.14)
    elif tp > 0.01:
        if tp < 1.0:
            nat = np.ones_like(t_vals)
            entry_zone = t_vals < 0.08
            exit_zone = t_vals > 0.86
            nat[entry_zone] = 0.65 + 0.35 * (t_vals[entry_zone] / 0.08)
            nat[exit_zone] = 1.0 - 0.50 * ((t_vals[exit_zone] - 0.86) / 0.14)
            taper_window = (1.0 - tp) * 1.0 + tp * nat
        else:
            whip_exp = 0.55 * (tp - 0.5)
            entry_ramp = np.minimum(1.0, 0.55 + 0.45 * (t_vals / 0.08))
            exit_whip = np.maximum(0.08, (1.0 - 0.88 * (t_vals ** 1.2)) ** whip_exp)
            taper_window = entry_ramp * exit_whip
    elif tp < -0.01:
        inv_exp = 0.65 * abs(tp)
        taper_window = np.maximum(0.10, (0.12 + 0.88 * t_vals) ** inv_exp)

    r_t = np.maximum(0.30, r_t * taper_window)
    d_t = np.clip(np.interp(t_vals, u_chord, densities), 0.05, 1.0)

    # Convert from pixel coordinates to physical canvas millimeters
    xs_mm = x_t * transform.scale_mm_per_px + transform.offset_x_mm
    ys_mm = y_t * transform.scale_mm_per_px + transform.offset_y_mm
    max_r_mm_cap = c_cfg.brush_max_radius_mm * max(1.0, p_cfg.stroke_width_scale)
    rs_mm = np.clip(r_t * transform.scale_mm_per_px, 0.1, max_r_mm_cap)

    # Analytic derivatives & curvature in mm space
    dx_mm = np.gradient(xs_mm, t_vals)
    dy_mm = np.gradient(ys_mm, t_vals)
    ddx_mm = np.gradient(dx_mm, t_vals)
    ddy_mm = np.gradient(dy_mm, t_vals)

    speed_sq = dx_mm * dx_mm + dy_mm * dy_mm
    speed = np.sqrt(np.maximum(speed_sq, 1e-10))
    denom = np.maximum(speed_sq ** 1.5, 1e-10)
    kappa_t = np.abs(dx_mm * ddy_mm - dy_mm * ddx_mm) / denom

    tx_t = dx_mm / speed
    ty_t = dy_mm / speed

    # Physical Brush-Drag Offset (Trailer Kinematics):
    alpha = c_cfg.brush_drag_coeff
    l_drag = alpha * rs_mm
    x_machine = xs_mm + l_drag * tx_t
    y_machine = ys_mm + l_drag * ty_t

    # Pressure (Z) Mapping
    r_eff = rs_mm * np.sqrt(d_t)
    a_poly, b_poly, c_poly = c_cfg.z_calibration_poly
    z_t = a_poly * (r_eff ** 2) + b_poly * r_eff + c_poly

    # 3-Phase Velocity (F) Profiling & Exit Whip Z Lift
    feed_t = np.empty_like(t_vals)
    entry_mask = t_vals <= 0.12
    mid_mask = (t_vals > 0.12) & (t_vals < 0.85)
    whip_mask = t_vals >= 0.85

    feed_t[entry_mask] = c_cfg.f_min_mm_min
    feed_t[mid_mask] = np.maximum(
        c_cfg.f_min_mm_min,
        c_cfg.f_cruise_mm_min / (1.0 + c_cfg.curvature_slowdown_gain * kappa_t[mid_mask]),
    )
    if np.any(whip_mask):
        whip_phase = (t_vals[whip_mask] - 0.85) / 0.15
        mid_indices = np.where(mid_mask)[0]
        f_whip_start = (
            float(feed_t[mid_indices[-1]]) if len(mid_indices) > 0 else c_cfg.f_cruise_mm_min
        )
        feed_t[whip_mask] = (1.0 - whip_phase) * f_whip_start + whip_phase * c_cfg.f_max_mm_min

        z_whip_start = float(z_t[mid_indices[-1]]) if len(mid_indices) > 0 else float(z_t[0])
        z_t[whip_mask] = (1.0 - whip_phase) * z_whip_start + whip_phase * c_cfg.z_clearance_mm
        z_t[-1] = c_cfg.z_clearance_mm

    init_ratio = float(np.clip(r_eff[0] / max(c_cfg.brush_max_radius_mm, 1e-3), 0.0, 1.0))
    dwell_start_ms = int(round(30.0 + 30.0 * init_ratio))

    stroke_rgb = (
        _jitter_rgb_hue(target_rgb, p_cfg.stroke_hue_jitter, stroke_id * 131 + layer_index * 17)
        if p_cfg.stroke_hue_jitter > 0.01
        else target_rgb
    )
    stroke_hex = _rgb_to_hex(stroke_rgb)

    waypoints = [
        Waypoint(
            t=round(float(t_vals[i]), 4),
            x_mm=round(float(x_machine[i]), 3),
            y_mm=round(float(y_machine[i]), 3),
            r_mm=round(float(rs_mm[i]), 3),
            z_mm=round(float(z_t[i]), 3),
            feed_mm_min=round(float(feed_t[i]), 1),
        )
        for i in range(num_samples)
    ]

    stroke = Stroke(
        stroke_id=stroke_id,
        stroke_type=path.stroke_type,
        layer_index=layer_index,
        layer_name=layer_name,
        color_hex=stroke_hex,
        dwell_start_ms=dwell_start_ms,
        waypoints=waypoints,
    )

    return ParameterizedStrokeData(
        stroke=stroke,
        raw_spline_xy_mm=np.column_stack([xs_mm, ys_mm]),
        machine_xy_mm=np.column_stack([x_machine, y_machine]),
        curvature=kappa_t,
    )


def run_multilayer_plotter_pipeline(
    image_path: str | Path,
    painterly_config: PainterlyConfig | None = None,
    calibration_config: PhysicalCalibrationConfig | None = None,
    fit_canvas_to_image_aspect: bool = True,
) -> StrokeVectorDocument:
    """Execute the K-layer acrylic plotter pipeline matching Painterly Multi-Layer decomposition.

    Decomposes the source image into the same K color/tonal layers as `run_painterly_pipeline`,
    subdivides wide regions per layer using the physical brush radius (`brush_max_radius_mm`)
    and stepover (`wash_Stepover_factor`), applies Stage 5 polarity resolution and Stage 6
    physical kinematics (`z_mm`, `feed_mm_min`, bristle drag offset), and runs banded TSP
    path ordering independently within each paint layer.
    """
    p_cfg = painterly_config or PainterlyConfig()
    c_cfg = calibration_config or PhysicalCalibrationConfig()

    # Reuse the Painterly pipeline's cached Stage 1-3 topology & palette extraction
    p_doc = run_painterly_pipeline(image_path, config=p_cfg)
    width_px, height_px = p_doc.width_px, p_doc.height_px
    num_layers = len(p_doc.layers)

    box_w_mm, box_h_mm = c_cfg.canvas_dims_mm
    scale_mm_per_px = min(box_w_mm / max(width_px, 1), box_h_mm / max(height_px, 1))
    if fit_canvas_to_image_aspect:
        eff_w_mm = round(width_px * scale_mm_per_px, 3)
        eff_h_mm = round(height_px * scale_mm_per_px, 3)
        transform = CanvasTransform(
            image_width_px=width_px,
            image_height_px=height_px,
            canvas_width_mm=eff_w_mm,
            canvas_height_mm=eff_h_mm,
            scale_mm_per_px=scale_mm_per_px,
            offset_x_mm=0.0,
            offset_y_mm=0.0,
        )
    else:
        transform = CanvasTransform.from_image_and_canvas(
            width_px=width_px,
            height_px=height_px,
            canvas_dims_mm=c_cfg.canvas_dims_mm,
        )

    img_path = Path(image_path).resolve()
    try:
        mtime_ns = img_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0

    topo_key = (
        str(img_path),
        mtime_ns,
        p_cfg.num_tonal_layers,
        p_cfg.color_palette_mode,
        tuple(p_cfg.layer_thresholds) if p_cfg.layer_thresholds else None,
        round(p_cfg.min_density_threshold, 4),
        round(p_cfg.max_density_threshold, 4),
        round(p_cfg.anisotropic_sigma, 4),
        round(p_cfg.flow_lock_strength, 4),
        round(p_cfg.flow_lock_angle_deg, 2),
        round(p_cfg.spur_prune_factor, 4),
        round(p_cfg.min_stroke_length_px, 4),
        round(p_cfg.max_deflection_deg, 4),
        p_cfg.normalize_paper_illumination,
    )
    cached_topo = _TOPOLOGY_CACHE.get(topo_key)
    per_layer_topo = cached_topo["per_layer_topo"] if cached_topo is not None else []

    res_scale = max(1.0, float(max(width_px, height_px)) / 480.0)
    eff_min_len_px = p_cfg.min_stroke_length_px * res_scale
    physical_r_max_px = max(1.5 * res_scale, transform.radius_mm_to_px(c_cfg.brush_max_radius_mm))
    stage_cfg = PhysicalCalibrationConfig(
        spur_prune_factor=p_cfg.spur_prune_factor,
        min_stroke_length_px=eff_min_len_px,
        max_deflection_deg=p_cfg.max_deflection_deg,
    )

    canvas_h_mm = transform.canvas_height_mm
    num_bands = max(1, c_cfg.num_spatial_bands)
    band_height_mm = max(canvas_h_mm / num_bands, 1.0)

    all_ordered_strokes: list[Stroke] = []
    plotter_layers: list[PlotterLayerInfo] = []
    total_air_before = 0.0
    total_air_after = 0.0
    global_stroke_id = 0

    for idx, p_layer in enumerate(p_doc.layers):
        layer_frac = idx / max(num_layers - 1, 1)
        # Scale physical max brush radius slightly smaller on upper detail layers to mimic fine brushwork
        layer_r_max_px = max(2.0 * res_scale, physical_r_max_px * (1.0 - 0.38 * layer_frac))
        stepover_px = max(1.5 * res_scale, c_cfg.wash_Stepover_factor * layer_r_max_px)

        if idx < len(per_layer_topo) and "untangled_paths" in per_layer_topo[idx]:
            untangled_paths: list[PixelPath] = per_layer_topo[idx]["untangled_paths"]
        else:
            s3 = run_stage3_graph_untangling(
                pruned_skeleton=p_layer.skeleton,
                edt_radius_px=p_layer.edt_radius_px,
                density_map=p_doc.density_map,
                config=stage_cfg,
            )
            untangled_paths = s3.untangled_paths

        # Subdivide wide regions (> layer_r_max_px) using physical brush stepover
        expanded_paths: list[PixelPath] = []
        for p in untangled_paths:
            eff_max_r = float(np.max(p.radii_px)) * p_cfg.stroke_width_scale
            if eff_max_r > layer_r_max_px * 1.12:
                subs = _subdivide_broad_wash(
                    path=p,
                    stroke_mask=p_layer.mask,
                    edt_radius_px=p_layer.edt_radius_px,
                    density_map=p_doc.density_map,
                    stepover_px=stepover_px,
                    r_max_px=layer_r_max_px,
                    min_length_px=eff_min_len_px,
                    start_id=len(expanded_paths),
                )
                if subs:
                    expanded_paths.extend(subs)
                else:
                    expanded_paths.append(p)
            else:
                callig_p = PixelPath(
                    path_id=len(expanded_paths),
                    coords_xy_px=p.coords_xy_px.copy(),
                    radii_px=np.minimum(p.radii_px, layer_r_max_px),
                    densities=p.densities.copy(),
                    stroke_type="calligraphic_centerline",
                )
                expanded_paths.append(callig_p)

        # Stage 5: Polarity Resolution (press -> lift)
        s5 = run_stage5_polarity_resolution(
            classified_paths=expanded_paths,
            density_map=p_doc.density_map,
            config=stage_cfg,
        )

        # Parameterize each stroke with Painterly geometry + Stage 6 physical kinematics
        layer_pdata: list[ParameterizedStrokeData] = []
        for path in s5.oriented_paths:
            pdata = _parameterize_plotter_layer_stroke(
                path=path,
                stroke_id=len(layer_pdata),
                layer_index=idx,
                layer_name=p_layer.layer_name,
                layer_max_radius_px=layer_r_max_px,
                target_rgb=p_layer.target_rgb,
                transform=transform,
                p_cfg=p_cfg,
                c_cfg=c_cfg,
                res_scale=res_scale,
            )
            if pdata is not None:
                layer_pdata.append(pdata)

        layer_air_before = _compute_air_travel([d.stroke for d in layer_pdata])
        total_air_before += layer_air_before

        # Per-Layer Banded TSP Path Ordering
        bands: dict[int, list[ParameterizedStrokeData]] = {b: [] for b in range(num_bands)}
        for item in layer_pdata:
            mean_y = float(np.mean([wp.y_mm for wp in item.stroke.waypoints]))
            b_idx = int(np.clip(math.floor(mean_y / band_height_mm), 0, num_bands - 1))
            bands[b_idx].append(item)

        ordered_layer_pdata: list[ParameterizedStrokeData] = []
        curr_pos = (0.0, 0.0)
        for b_idx in range(num_bands):
            if not bands[b_idx]:
                continue
            band_ordered = _solve_band_tsp(bands[b_idx], curr_pos)
            ordered_layer_pdata.extend(band_ordered)
            curr_pos = _stroke_end_xy(band_ordered[-1].stroke)

        layer_strokes: list[Stroke] = []
        for item in ordered_layer_pdata:
            item.stroke.stroke_id = global_stroke_id
            global_stroke_id += 1
            layer_strokes.append(item.stroke)
            all_ordered_strokes.append(item.stroke)

        layer_air_after = _compute_air_travel(layer_strokes)
        total_air_after += layer_air_after

        plotter_layers.append(
            PlotterLayerInfo(
                layer_index=idx,
                layer_name=p_layer.layer_name,
                density_threshold=p_layer.density_threshold,
                target_gray=p_layer.target_gray,
                color_hex=p_layer.color_hex,
                auto_color_hex=p_layer.auto_color_hex,
                stroke_count=len(layer_strokes),
                air_travel_mm=round(layer_air_after, 2),
            )
        )

    return StrokeVectorDocument(
        canvas_dims_mm=(transform.canvas_width_mm, transform.canvas_height_mm),
        source_dims_px=(width_px, height_px),
        stroke_count=len(all_ordered_strokes),
        strokes=all_ordered_strokes,
        paper_hex=p_doc.paper_hex,
        layers=plotter_layers,
        air_travel_before_mm=round(total_air_before, 2),
        air_travel_after_mm=round(total_air_after, 2),
    )
