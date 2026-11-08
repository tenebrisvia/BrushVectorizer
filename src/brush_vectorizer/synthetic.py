"""Synthetic multi-color landscape artwork generator for the Web Studio and unit tests."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def generate_synthetic_color_artwork(
    width_px: int = 520,
    height_px: int = 620,
    seed: int = 42,
) -> np.ndarray:
    """Create a realistic painterly alpine landscape (blue sky, white mountain, green hills/pines).

    Returns an 8-bit RGB array of shape (height_px, width_px, 3).
    Designed with expressive brush bands and ridge geometry so both the source image and its
    multi-layer vectorization capture sky washes, snow-white peaks, slate mountain shadows,
    rolling green meadows, and foreground evergreen trees.
    """
    rng = np.random.default_rng(seed)
    # Crisp snow-white / warm alpine paper base (serves as bright snow faces & cloud highlights)
    canvas = np.full((height_px, width_px, 3), (247.0, 248.0, 250.0), dtype=np.float64)
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    sx = width_px / 520.0
    sy = height_px / 620.0

    def _stamp_stroke(
        pts_xy: np.ndarray,
        radii: np.ndarray,
        rgb: tuple[float, float, float],
        alpha_profile: np.ndarray,
        softness: float = 0.45,
    ) -> None:
        nonlocal canvas
        n = len(pts_xy)
        target = np.array(rgb, dtype=np.float64).reshape((1, 1, 3))
        for i in range(n):
            cx, cy = pts_xy[i]
            r = max(float(radii[i]), 0.8)
            a_val = float(alpha_profile[i])
            r_pad = int(np.ceil(r + 2))
            y0, y1 = max(0, int(cy) - r_pad), min(height_px, int(cy) + r_pad + 1)
            x0, x1 = max(0, int(cx) - r_pad), min(width_px, int(cx) + r_pad + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            dist = np.hypot(xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy)
            mask = dist <= r
            if not np.any(mask):
                continue
            norm_d = dist / r
            falloff = np.clip(1.0 - softness * (norm_d ** 2), 0.0, 1.0)
            alpha = (a_val * falloff)[:, :, None]
            patch = canvas[y0:y1, x0:x1]
            patch[mask] = (patch * (1.0 - alpha) + target * alpha)[mask]

    # Mountain silhouette profile y_mtn(x): twin alpine peaks (main peak at x=290, secondary at x=135)
    x_grid = np.arange(width_px, dtype=np.float64)
    xn = x_grid / max(width_px, 1)
    mtn_ridge_y = (
        275.0 * sy
        - 155.0 * sy * np.exp(-((xn - 0.56) / 0.19) ** 2)
        - 95.0 * sy * np.exp(-((xn - 0.25) / 0.14) ** 2)
        - 45.0 * sy * np.exp(-((xn - 0.82) / 0.12) ** 2)
        + 6.0 * sy * np.sin(xn * 28.0)
    )

    # 1. BLUE SKY: Sweeping cerulean & cobalt sky bands above the mountain silhouette,
    #    leaving soft white cloud openings and crisp white snow along the mountain crest
    sky_bands = [
        (34.0, (48.0, 118.0, 210.0), 22.0, 0.78),
        (68.0, (58.0, 132.0, 222.0), 21.0, 0.72),
        (104.0, (74.0, 148.0, 230.0), 19.0, 0.66),
        (142.0, (92.0, 165.0, 236.0), 18.0, 0.58),
        (180.0, (112.0, 180.0, 240.0), 16.0, 0.50),
        (215.0, (130.0, 192.0, 242.0), 15.0, 0.44),
    ]
    for idx, (base_y, sky_rgb, base_r, base_a) in enumerate(sky_bands):
        t = np.linspace(0.0, 1.0, 150)
        xs = (22.0 + 476.0 * t) * sx
        ys = (base_y + 7.0 * np.sin(2.0 * np.pi * t + idx * 0.8)) * sy
        # Clip sky strokes so they stay above the white mountain snow crest
        mtn_at_x = np.interp(xs, x_grid, mtn_ridge_y)
        keep = ys < (mtn_at_x - 10.0 * sy)
        # Also leave a sunlit white cloud gap in the middle-left sky on rows 2-3
        if idx in (2, 3):
            cloud_gap = (t > 0.14) & (t < 0.36)
            keep = keep & (~cloud_gap)
        padded = np.concatenate([[False], keep, [False]])
        starts = np.where(padded[1:] & ~padded[:-1])[0]
        ends = np.where(~padded[1:] & padded[:-1])[0]
        for s0, e0 in zip(starts, ends):
            if e0 - s0 < 5:
                continue
            seg_t = np.linspace(0.0, 1.0, e0 - s0)
            seg_r = (base_r * (0.85 + 0.25 * np.sin(np.pi * seg_t))) * min(sx, sy)
            seg_a = np.full(e0 - s0, base_a)
            _stamp_stroke(np.column_stack([xs[s0:e0], ys[s0:e0]]), seg_r, sky_rgb, seg_a)

    # 2. WHITE MOUNTAIN PEAKS & GLACIER SHADOW RIDGES:
    #    The sunlit left slopes remain bright snow-white (canvas tone), while the right slopes,
    #    arêtes, and glacial valleys are sculpted with slate-blue and cool stone brushstrokes.
    #    2a. Cool glacier-blue shadow washes on the right flanks of both peaks
    glacier_strokes = [
        # Main peak right shadow face (from summit (291, 120) down-right)
        ((294.0, 124.0), (415.0, 275.0), 18.0, (118.0, 146.0, 182.0), 0.68),
        ((305.0, 165.0), (445.0, 300.0), 20.0, (102.0, 130.0, 168.0), 0.64),
        ((280.0, 195.0), (380.0, 305.0), 16.0, (128.0, 155.0, 188.0), 0.58),
        # Secondary left peak right shadow face (from summit (130, 180) down-right into saddle)
        ((132.0, 182.0), (225.0, 285.0), 15.0, (122.0, 150.0, 184.0), 0.65),
        ((148.0, 215.0), (240.0, 305.0), 16.0, (108.0, 136.0, 172.0), 0.58),
        # Far right peak shadow flank
        ((426.0, 232.0), (492.0, 305.0), 14.0, (110.0, 138.0, 174.0), 0.62),
    ]
    for (x0, y0), (x1, y1), rad, col_rgb, alpha_v in glacier_strokes:
        t = np.linspace(0.0, 1.0, 90)
        px = (x0 + (x1 - x0) * t + 6.0 * np.sin(np.pi * t)) * sx
        py = (y0 + (y1 - y0) * t) * sy
        pr = (rad * (1.0 - 0.35 * t)) * min(sx, sy)
        pa = np.full_like(t, alpha_v)
        _stamp_stroke(np.column_stack([px, py]), pr, col_rgb, pa)

    #    2b. Crisp alpine ridge outlines & dark slate crags framing the bright white snow faces
    ridge_lines = [
        # Left sunlit snow ridge of main peak (thin slate-blue contour defining the white snow edge)
        ((195.0, 255.0), (291.0, 120.0), 4.2, (74.0, 98.0, 132.0), 0.78),
        # Summit crag & central arête descending from main peak
        ((291.0, 120.0), (318.0, 285.0), 5.5, (58.0, 76.0, 106.0), 0.82),
        ((291.0, 120.0), (418.0, 268.0), 5.0, (64.0, 84.0, 115.0), 0.80),
        # Left sunlit ridge of secondary peak
        ((42.0, 272.0), (130.0, 180.0), 4.0, (78.0, 102.0, 136.0), 0.75),
        ((130.0, 180.0), (205.0, 255.0), 4.8, (62.0, 82.0, 112.0), 0.80),
        # Rocky crevices inside the snow face
        ((252.0, 178.0), (225.0, 268.0), 3.6, (88.0, 112.0, 145.0), 0.65),
        ((342.0, 185.0), (365.0, 280.0), 4.2, (52.0, 68.0, 96.0), 0.78),
    ]
    for (x0, y0), (x1, y1), rad, col_rgb, alpha_v in ridge_lines:
        t = np.linspace(0.0, 1.0, 90)
        px = (x0 + (x1 - x0) * t + 3.5 * np.sin(2.0 * np.pi * t)) * sx
        py = (y0 + (y1 - y0) * t) * sy
        pr = (rad * (1.1 - 0.45 * t)) * min(sx, sy)
        pa = alpha_v * (1.0 - 0.2 * (t ** 2))
        _stamp_stroke(np.column_stack([px, py]), pr, col_rgb, pa)

    # 3. GREEN LANDSCAPE: Rolling alpine meadows, emerald hills & deep forest ridges
    #    3a. Sunlit warm meadow-green & sage foothills (y = 305..450)
    hill_washes = [
        (318.0, -18.0, (112.0, 168.0, 88.0), 21.0, 0.72),
        (348.0, 22.0, (92.0, 154.0, 76.0), 22.0, 0.76),
        (382.0, -25.0, (76.0, 140.0, 68.0), 22.0, 0.80),
        (418.0, 20.0, (64.0, 126.0, 60.0), 23.0, 0.82),
        (456.0, -18.0, (52.0, 112.0, 54.0), 23.0, 0.84),
        (496.0, 15.0, (44.0, 98.0, 48.0), 24.0, 0.86),
        (536.0, -12.0, (38.0, 86.0, 44.0), 22.0, 0.88),
        (572.0, 8.0, (34.0, 76.0, 40.0), 20.0, 0.88),
    ]
    for idx, (y_base, wave_amp, green_rgb, rad, alpha_v) in enumerate(hill_washes):
        t = np.linspace(0.0, 1.0, 160)
        xs = (20.0 + 480.0 * t) * sx
        ys = (
            y_base
            + wave_amp * np.sin(1.4 * np.pi * t + idx * 0.55)
            + 8.0 * np.cos(3.0 * np.pi * t)
        ) * sy
        rs = (rad * (0.9 + 0.25 * np.sin(np.pi * t))) * min(sx, sy)
        al = np.full_like(t, alpha_v)
        _stamp_stroke(np.column_stack([xs, ys]), rs, green_rgb, al)

    # 4. FOREGROUND PINE TREES & DARK EVERGREEN CALLIGRAPHIC ACCENTS:
    #    Provide crisp high-density foreground structure for the darkest tonal layers
    pine_trees = [
        (78.0, 565.0, 395.0, 5.2),
        (122.0, 575.0, 430.0, 4.4),
        (412.0, 568.0, 380.0, 5.6),
        (456.0, 580.0, 425.0, 4.6),
    ]
    for tx_pos, y_bot, y_top, trunk_r in pine_trees:
        # Vertical trunk / core spine
        t = np.linspace(0.0, 1.0, 80)
        px = (tx_pos + 2.0 * np.sin(np.pi * t)) * sx
        py = (y_bot + (y_top - y_bot) * t) * sy
        pr = (trunk_r * (1.0 - 0.65 * t)) * min(sx, sy)
        _stamp_stroke(
            np.column_stack([px, py]),
            pr,
            (22.0, 38.0, 28.0),
            np.full_like(t, 0.92),
        )
        # Sweeping pine boughs across the trunk
        for b_frac in np.linspace(0.18, 0.88, 6):
            by = y_bot + (y_top - y_bot) * b_frac
            span = (1.0 - 0.72 * b_frac) * 32.0
            tb = np.linspace(0.0, 1.0, 36)
            bx = (tx_pos - span + 2.0 * span * tb) * sx
            b_curve = (by - 6.0 + 10.0 * np.abs(tb - 0.5) * 2.0) * sy
            br = (3.8 * (1.0 - 0.4 * b_frac) * (1.0 - 0.5 * np.abs(tb - 0.5))) * min(sx, sy)
            _stamp_stroke(
                np.column_stack([bx, b_curve]),
                br,
                (20.0, 46.0, 30.0),
                np.full_like(tb, 0.90),
            )

    noise = rng.normal(0.0, 1.2, size=canvas.shape)
    return np.clip(np.round(canvas + noise), 0, 255).astype(np.uint8)


def save_rgb_image(rgb: np.ndarray, output_path: str | Path) -> Path:
    """Save an 8-bit RGB array (H, W, 3) as a PNG image."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(out)
    return out
