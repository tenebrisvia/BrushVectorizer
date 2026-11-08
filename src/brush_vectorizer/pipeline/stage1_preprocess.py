"""Stage 1: Preprocessing & Tonal Segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

from brush_vectorizer.config import PhysicalCalibrationConfig


@dataclass
class Stage1Result:
    """Outputs of Stage 1 Preprocessing & Tonal Segmentation."""

    luminance_map: np.ndarray        # Y(x, y) in [0.0, 1.0], float64
    density_map: np.ndarray          # Unblurred Optical Density D(x, y) = 1.0 - Y(x, y)
    bridged_density_map: np.ndarray  # Anisotropically bridged density map
    stroke_mask: np.ndarray          # Binary mask M_stroke (bool)
    threshold_used: float            # Effective threshold value applied


def load_luminance_image(image_path: str | Path) -> np.ndarray:
    """Load an 8-bit or 16-bit grayscale/RGB/RGBA image into normalized luminance Y in [0, 1].

    1.0 = paper/background (white), 0.0 = maximum ink/charcoal density (black).
    """
    img = Image.open(image_path)
    arr = np.asarray(img)

    if arr.dtype == np.uint16:
        scale = 65535.0
    elif arr.dtype == np.uint8:
        scale = 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        scale = 255.0 if max_val > 1.0 else 1.0
    else:
        arr = arr.astype(np.float64)
        scale = float(np.max(arr)) if np.max(arr) > 0 else 1.0

    arr_f = arr.astype(np.float64) / scale
    arr_f = np.clip(arr_f, 0.0, 1.0)

    if arr_f.ndim == 2:
        return arr_f
    if arr_f.ndim == 3:
        if arr_f.shape[2] == 4:
            # Composite RGBA over white paper background (1.0)
            alpha = arr_f[:, :, 3:4]
            rgb = arr_f[:, :, :3] * alpha + (1.0 - alpha)
        elif arr_f.shape[2] == 3:
            rgb = arr_f[:, :, :3]
        elif arr_f.shape[2] == 2:
            # Grayscale + Alpha
            alpha = arr_f[:, :, 1]
            return np.clip(arr_f[:, :, 0] * alpha + (1.0 - alpha), 0.0, 1.0)
        else:
            return arr_f[:, :, 0]
        # ITU-R BT.709 luminance
        luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        return np.clip(luminance, 0.0, 1.0)

    raise ValueError(f"Unsupported image array shape: {arr.shape}")


def compute_otsu_threshold(values: np.ndarray, num_bins: int = 256) -> float:
    """Compute Otsu's threshold on normalized [0, 1] floating-point array."""
    flat = values.ravel()
    if flat.size == 0 or np.allclose(flat.min(), flat.max()):
        return 0.15

    hist, bin_edges = np.histogram(flat, bins=num_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = float(hist.sum())
    if total == 0:
        return 0.15

    weight1 = np.cumsum(hist) / total
    weight2 = 1.0 - weight1

    mean1_num = np.cumsum(hist * bin_centers)
    total_mean = mean1_num[-1]

    valid = (weight1 > 1e-6) & (weight2 > 1e-6)
    if not np.any(valid):
        return 0.15

    mean1 = np.zeros_like(bin_centers)
    mean2 = np.zeros_like(bin_centers)
    mean1[valid] = mean1_num[valid] / (weight1[valid] * total)
    mean2[valid] = (total_mean - mean1_num[valid]) / (weight2[valid] * total)

    between_var = weight1 * weight2 * (mean1 - mean2) ** 2
    best_idx = int(np.argmax(between_var))
    return float(bin_centers[best_idx])


def anisotropic_gap_bridging(
    density: np.ndarray,
    sigma: float = 1.0,
    flow_lock_strength: float = 0.0,
    flow_lock_angle_deg: float = 45.0,
) -> np.ndarray:
    """Apply structure-tensor-guided directional smoothing and morphological gap bridging.

    Bridges high-frequency 'flying white' dry-brush striations along the dominant stroke
    tangent while keeping perpendicular stroke boundaries sharp. Optionally overrides
    natural gradient flow with a forced directional closing angle (flow_lock_strength > 0).
    """
    if sigma <= 0.0 and flow_lock_strength <= 0.01:
        return density.copy()

    eff_sigma = max(0.3, sigma) if sigma > 0.0 else 0.6

    # 1. Isotropic baseline Gaussian blur (sigma ~ 1.0 px as specified in Stage 1.2)
    iso_blurred = ndimage.gaussian_filter(density, sigma=eff_sigma) if sigma > 0.0 else density.copy()

    # 2. Compute structure tensor J = G_rho * (grad D * grad D^T)
    gx = ndimage.gaussian_filter(density, sigma=max(0.8, eff_sigma), order=[0, 1])
    gy = ndimage.gaussian_filter(density, sigma=max(0.8, eff_sigma), order=[1, 0])

    rho = max(1.5, 2.0 * eff_sigma)
    j_xx = ndimage.gaussian_filter(gx * gx, sigma=rho)
    j_xy = ndimage.gaussian_filter(gx * gy, sigma=rho)
    j_yy = ndimage.gaussian_filter(gy * gy, sigma=rho)

    # Dominant gradient angle and tangent angle (perpendicular to gradient)
    # Tangent vector points along the stroke striations
    grad_angle = 0.5 * np.arctan2(2.0 * j_xy, j_xx - j_yy)
    tangent_angle = grad_angle + 0.5 * np.pi

    # Coherence measures anisotropy: ((lambda1 - lambda2) / (lambda1 + lambda2))^2
    trace = j_xx + j_yy
    det_diff = np.hypot(j_xx - j_yy, 2.0 * j_xy)
    coherence = np.where(trace > 1e-6, (det_diff / (trace + 1e-8)) ** 2, 0.0)

    # 3. Quantize tangent orientations into 4 directional closing kernels (0, 45, 90, 135 deg)
    # to bridge dry-brush striation gaps along the local stroke flow
    angles_deg = (np.rad2deg(tangent_angle) % 180.0)
    if flow_lock_strength > 0.01:
        target_deg = float(flow_lock_angle_deg) % 180.0
        angles_deg = (1.0 - flow_lock_strength) * angles_deg + flow_lock_strength * target_deg
        coherence = np.maximum(coherence, flow_lock_strength)

    # Build 3x3 directional line structuring elements
    k_0 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)
    k_45 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool)
    k_90 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool)
    k_135 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)

    c_0 = ndimage.grey_closing(iso_blurred, footprint=k_0)
    c_45 = ndimage.grey_closing(iso_blurred, footprint=k_45)
    c_90 = ndimage.grey_closing(iso_blurred, footprint=k_90)
    c_135 = ndimage.grey_closing(iso_blurred, footprint=k_135)

    # Select directional closing matching local tangent orientation
    sector = np.floor(((angles_deg + 22.5) % 180.0) / 45.0).astype(int)
    directional_closed = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3],
        [c_0, c_135, c_90, c_45],
        default=iso_blurred,
    )

    # Blend isotropic blur and directional closing weighted by local anisotropy coherence
    bridged = (1.0 - 0.5 * coherence) * iso_blurred + (0.5 * coherence) * directional_closed

    # When flow_lock_strength > 0.15, apply an oriented line kernel along flow_lock_angle_deg
    # to fuse disconnected marks into scanline/rain/directional streaks
    if flow_lock_strength > 0.15:
        rad = np.deg2rad(float(flow_lock_angle_deg) % 180.0)
        half_len = max(1, int(round(1.0 + 4.0 * flow_lock_strength)))
        ksize = 2 * half_len + 1
        footprint = np.zeros((ksize, ksize), dtype=bool)
        for step in np.linspace(-half_len, half_len, ksize * 2 + 1):
            rx = int(round(half_len + step * np.cos(rad)))
            ry = int(round(half_len - step * np.sin(rad)))
            if 0 <= ry < ksize and 0 <= rx < ksize:
                footprint[ry, rx] = True
        locked_closed = ndimage.grey_closing(bridged, footprint=footprint)
        bridged = (1.0 - 0.75 * flow_lock_strength) * bridged + (0.75 * flow_lock_strength) * locked_closed

    return np.clip(bridged, 0.0, 1.0)


def normalize_paper_luminance(luminance: np.ndarray) -> np.ndarray:
    """Normalize raw luminance Y so the paper background maps to 1.0 and black ink to 0.0.

    Handles photographs or scans of warm/cream paper and uneven lighting gradients
    without erasing broad charcoal washes.
    """
    lum = np.clip(luminance.astype(np.float64), 0.0, 1.0)
    h, w = lum.shape
    if h < 4 or w < 4:
        return lum

    p50 = float(np.percentile(lum, 50))
    p85 = float(np.percentile(lum, 85))
    if p85 <= 0.05:
        return lum

    # Estimate low-frequency paper illumination surface from background-candidate pixels.
    # Downsample before running large-sigma Gaussian blur for ~300x speedup on high-res images.
    paper_candidates = lum >= p50
    sigma_bg = max(float(max(h, w)) / 6.0, 8.0)
    ds = max(1, min(16, int(round(sigma_bg / 10.0))))

    if ds > 1 and h >= ds * 4 and w >= ds * 4:
        lum_masked = np.where(paper_candidates, lum, 0.0)[::ds, ::ds]
        mask_sub = paper_candidates[::ds, ::ds].astype(np.float64)
        sigma_sub = sigma_bg / float(ds)
        weighted_lum_sub = ndimage.gaussian_filter(lum_masked, sigma=sigma_sub)
        weight_mask_sub = ndimage.gaussian_filter(mask_sub, sigma=sigma_sub)
        paper_sub = np.where(weight_mask_sub > 1e-4, weighted_lum_sub / weight_mask_sub, p85)
        paper_sub = np.maximum(paper_sub, max(0.35, 0.85 * p85))

        # Bilinearly upsample smooth illumination field back to exact (h, w)
        zy = h / float(paper_sub.shape[0])
        zx = w / float(paper_sub.shape[1])
        paper_field = ndimage.zoom(paper_sub, (zy, zx), order=1)
        paper_field = paper_field[:h, :w]
        if paper_field.shape != (h, w):
            pad_h = max(0, h - paper_field.shape[0])
            pad_w = max(0, w - paper_field.shape[1])
            paper_field = np.pad(paper_field, ((0, pad_h), (0, pad_w)), mode="edge")[:h, :w]
        paper_field = np.maximum(paper_field, max(0.35, 0.85 * p85))
    else:
        weighted_lum = ndimage.gaussian_filter(
            np.where(paper_candidates, lum, 0.0), sigma=sigma_bg
        )
        weight_mask = ndimage.gaussian_filter(
            paper_candidates.astype(np.float64), sigma=sigma_bg
        )
        paper_field = np.where(weight_mask > 1e-4, weighted_lum / weight_mask, p85)
        paper_field = np.maximum(paper_field, max(0.35, 0.85 * p85))

    lum_normalized = lum / paper_field
    return np.clip(lum_normalized, 0.0, 1.0)


def run_stage1_preprocessing(
    luminance: np.ndarray,
    config: PhysicalCalibrationConfig,
) -> Stage1Result:
    """Execute Stage 1: Inversion & Normalization, Anisotropic Gap Bridging, and Thresholding."""
    # 1. Inversion & Normalization: normalize paper background to 1.0, then D(x, y) = 1.0 - Y(x, y)
    luminance_normalized = normalize_paper_luminance(luminance)
    density_map = 1.0 - luminance_normalized

    # 2. Anisotropic Gap Bridging (while keeping unblurred density_map intact)
    bridged_density = anisotropic_gap_bridging(density_map, sigma=config.anisotropic_sigma)

    # 3. Adaptive Thresholding
    otsu_val = compute_otsu_threshold(bridged_density)
    if config.threshold_mode == "fixed":
        threshold = config.density_floor
    elif config.threshold_mode == "otsu":
        threshold = max(otsu_val, 0.08)
    else:
        # 'hybrid': use density_floor (0.15) on background-normalized density,
        # or a scaled Otsu threshold when background noise is higher
        bg_noise_p80 = float(np.percentile(bridged_density, 80))
        safe_floor = max(config.density_floor, min(otsu_val, bg_noise_p80 + 0.10))
        threshold = min(otsu_val, safe_floor) if otsu_val >= config.density_floor else otsu_val

    stroke_mask = bridged_density > threshold

    # Clean isolated single-pixel speckle noise and tiny 1px pinholes inside strokes
    if np.any(stroke_mask):
        stroke_mask = ndimage.binary_closing(stroke_mask, structure=np.ones((3, 3), dtype=bool))
        # Remove tiny isolated 1-2 pixel dust specks
        stroke_mask = ndimage.binary_opening(
            stroke_mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        ) | (bridged_density > max(threshold + 0.10, otsu_val))

    return Stage1Result(
        luminance_map=luminance_normalized,
        density_map=density_map,
        bridged_density_map=bridged_density,
        stroke_mask=stroke_mask.astype(bool),
        threshold_used=float(threshold),
    )
