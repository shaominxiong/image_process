#!/usr/bin/env python3
"""Detect the fiducial cross marks in a DNG frame.

This module does one thing: locate the fiducial pattern and report where the marks
are. It has no plotting and no ROI logic -- see process_images.py for defining
regions of interest from the marks, extracting profiles, and plotting.

The targets are *dark* crosses printed on a brightly lit surface, arranged as two
concentric four-fold symmetric rings:

    *           *          <- outer ring (faint, wide square)
        +   +
                           <- pattern centre
        +   +
    *           *          <- inner ring (bold, narrow square)

Pipeline:

1. Flatten the illumination gradient with a large Gaussian high-pass so the marks
   become positive-going residuals independent of the LED falloff.
2. Score every pixel for "crossness": all four half-arms must be dark *and* the
   region just beyond each arm tip must be blank. The arm-termination term is what
   rejects the long ruler lines and the T-junction, which are otherwise perfect
   matches for a cross template.
3. Identify the inner ring as the best four-fold symmetric quad among the strong
   peaks, then find the outer ring by searching the weaker candidates for a quad
   concentric with it. The faint outer marks score no higher than dashed-line
   fragments, so geometric consensus (not response strength) is what finds them.
4. Refine every mark to sub-pixel accuracy with a residual-weighted centroid.

Frames without the pattern (e.g. the wide-field cam_1 views) are reported as "not
found" rather than fitted to spurious peaks: marks must clear a noise-relative
significance floor *and* form a symmetric quad.

Run standalone to dump mark coordinates as JSON:

    python detect_fiducials.py <file-or-dir.dng> [--output-dir output]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rawpy
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter

# Tuned for the 2592x1944 sensor frames in data/ (marks are ~40 px across).
DEFAULT_ARM_LENGTH = 20
DEFAULT_BAR_WIDTH = 7
DEFAULT_BACKGROUND_SIGMA = 25.0
# Real marks score ~8x the residual noise; frames without the pattern peak below 0.5x.
DEFAULT_MIN_SNR = 3.0
# Physical ring radii of the fiducial target, in mm. Only their *ratio* is used for
# geometry: it is scale-invariant, so it holds across batches shot at different
# distances, whereas mm-per-px does not and is recomputed per frame.
INNER_RADIUS_MM = 2.2
OUTER_RADIUS_MM = 6.0
NOMINAL_RING_RATIO = OUTER_RADIUS_MM / INNER_RADIUS_MM
# Measured on the 45 cm cam_2 batch: 2.6995 +- 0.0038, i.e. ~1% below nominal. Part
# of that is sub-pixel window bias on the closely spaced inner marks; the rest looks
# like the target differing from its nominal spec. Track ring_ratio_measured across
# batches to tell those apart.
# Guided outer search: window around each predicted position, and the (lower)
# significance floor allowed there because the position is strongly constrained.
DEFAULT_GUIDED_SEARCH_PX = 35
DEFAULT_GUIDED_MIN_SNR = 1.5


# --------------------------------------------------------------------------- I/O


def load_image(path: Path) -> np.ndarray:
    """Load a DNG file and return a grayscale image array."""
    raw = rawpy.imread(str(path))
    try:
        rgb = raw.postprocess(
            output_color=rawpy.ColorSpace.sRGB,
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True,
        )
    except TypeError:
        rgb = raw.postprocess()
    raw.close()

    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim == 3:
        return image.mean(axis=2)
    return image


def discover_images(input_path: Path, pattern: Optional[str] = None) -> List[Path]:
    """Resolve a file or directory argument to a sorted list of DNG paths.

    ``pattern`` filters a directory by filename glob (e.g. ``*cam_2*``), which is
    how a single camera is selected out of a mixed capture directory.
    """
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        if pattern:
            return sorted(p for p in input_path.glob(pattern) if p.suffix.lower() == ".dng")
        return sorted(input_path.glob("*.dng")) + sorted(input_path.glob("*.DNG"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


# ------------------------------------------------------------------- mark scoring


def _box_mean(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Separable box mean over a (height, width) footprint."""
    return uniform_filter(uniform_filter(array, size=(height, 1)), size=(1, width))


def _shift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift an array by (dy, dx), filling vacated cells with zero.

    Zero fill is deliberate: it drives the arm terms to zero near the border, so
    edge pixels can never win the min-over-arms test.
    """
    out = np.zeros_like(array)
    ys_dst = slice(max(dy, 0), array.shape[0] + min(dy, 0))
    xs_dst = slice(max(dx, 0), array.shape[1] + min(dx, 0))
    ys_src = slice(max(-dy, 0), array.shape[0] + min(-dy, 0))
    xs_src = slice(max(-dx, 0), array.shape[1] + min(-dx, 0))
    out[ys_dst, xs_dst] = array[ys_src, xs_src]
    return out


def high_pass(image: np.ndarray, sigma: float = DEFAULT_BACKGROUND_SIGMA) -> np.ndarray:
    """Remove the illumination gradient; dark marks become positive residuals."""
    return gaussian_filter(image, sigma) - image


def crossness_response(
    residual: np.ndarray,
    arm_length: int = DEFAULT_ARM_LENGTH,
    bar_width: int = DEFAULT_BAR_WIDTH,
) -> np.ndarray:
    """Score each pixel as the centre of an isolated dark cross.

    ``min`` over the four half-arms demands a complete cross; subtracting the
    ``max`` over the four beyond-the-tip patches demands that the arms actually
    stop, which is what distinguishes a fiducial from a ruler line or junction.
    """
    ext_length = max(8, int(round(arm_length * 1.2)))

    arm_h = _box_mean(residual, bar_width, arm_length)
    arm_v = _box_mean(residual, arm_length, bar_width)
    ext_h = _box_mean(residual, bar_width, ext_length)
    ext_v = _box_mean(residual, ext_length, bar_width)

    arm_offset = arm_length // 2 + 1
    ext_offset = arm_length + ext_length // 2 + 1

    arms = [
        _shift(arm_h, 0, -arm_offset),
        _shift(arm_h, 0, arm_offset),
        _shift(arm_v, -arm_offset, 0),
        _shift(arm_v, arm_offset, 0),
    ]
    extensions = [
        _shift(ext_h, 0, -ext_offset),
        _shift(ext_h, 0, ext_offset),
        _shift(ext_v, -ext_offset, 0),
        _shift(ext_v, ext_offset, 0),
    ]

    arm_min = np.minimum.reduce(arms)
    ext_max = np.maximum.reduce(extensions)
    return arm_min - np.maximum(ext_max, 0.0)


def find_peaks(
    response: np.ndarray,
    min_separation: int = 41,
    max_peaks: int = 400,
    border: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return non-maximum-suppressed peak coordinates (x, y) and their scores."""
    local_max = response == maximum_filter(response, size=min_separation)
    local_max[:border, :] = False
    local_max[-border:, :] = False
    local_max[:, :border] = False
    local_max[:, -border:] = False
    local_max &= response > 0.0

    ys, xs = np.nonzero(local_max)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)

    scores = response[ys, xs]
    order = np.argsort(scores)[::-1][:max_peaks]
    points = np.column_stack([xs[order], ys[order]]).astype(np.float64)
    return points, scores[order]


def refine_subpixel(residual: np.ndarray, point: Sequence[float], half_window: int = 14) -> Tuple[float, float]:
    """Refine a mark centre with a residual-weighted centroid."""
    x0, y0 = int(round(point[0])), int(round(point[1]))
    y_lo, y_hi = max(0, y0 - half_window), min(residual.shape[0], y0 + half_window + 1)
    x_lo, x_hi = max(0, x0 - half_window), min(residual.shape[1], x0 + half_window + 1)

    patch = residual[y_lo:y_hi, x_lo:x_hi]
    weights = np.clip(patch - np.median(patch), 0.0, None)
    total = weights.sum()
    if total <= 0:
        return float(x0), float(y0)

    ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    return float((weights * xs).sum() / total), float((weights * ys).sum() / total)


# ------------------------------------------------------------------- ring geometry


def _quad_symmetry_error(quad: np.ndarray, center: np.ndarray) -> Tuple[float, float]:
    """Return (relative radius spread, worst angular gap error) for four points."""
    offsets = quad - center
    radii = np.linalg.norm(offsets, axis=1)
    if radii.mean() <= 0:
        return np.inf, np.inf
    radius_spread = float((radii.max() - radii.min()) / radii.mean())

    angles = np.sort(np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0])) % 360.0)
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    return radius_spread, float(np.abs(gaps - 90.0).max())


def find_symmetric_quad(
    candidates: np.ndarray,
    center: Optional[np.ndarray] = None,
    min_radius: float = 0.0,
    max_radius: float = np.inf,
    radius_tolerance: float = 0.25,
    angle_tolerance_deg: float = 22.0,
    limit: int = 16,
) -> Optional[np.ndarray]:
    """Find four candidates forming a four-fold symmetric quad, ordered CCW.

    With ``center=None`` each combination is tested about its own centroid, which
    is how the inner ring is identified without any prior. With an explicit
    ``center`` the quad must also be concentric with it -- that is what pulls the
    faint outer ring out of a candidate pool containing dashed-line fragments of
    equal response strength.

    Returns None when no set of four candidates is consistent enough.
    """
    if len(candidates) < 4:
        return None

    pool = candidates
    if center is not None:
        radii = np.linalg.norm(candidates - center, axis=1)
        pool = candidates[(radii >= min_radius) & (radii <= max_radius)]
        if len(pool) < 4:
            return None

    # Candidates arrive strongest-first, so a bounded pool keeps the search cheap.
    pool = pool[:limit]
    best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None

    for combo in itertools.combinations(range(len(pool)), 4):
        quad = pool[list(combo)]
        quad_center = quad.mean(axis=0) if center is None else center
        radius_spread, angle_error = _quad_symmetry_error(quad, quad_center)
        if radius_spread > radius_tolerance or angle_error > angle_tolerance_deg:
            continue

        cost = radius_spread + angle_error / 90.0
        if best is None or cost < best[0]:
            best = (cost, quad, quad_center)

    if best is None:
        return None
    _, quad, quad_center = best
    return sort_points_by_angle(quad, (float(quad_center[0]), float(quad_center[1])))


def fit_circle(points: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """Fit a circle to 2D points via the algebraic (Kasa) method."""
    if len(points) < 3:
        raise ValueError("At least 3 points are required to fit a circle")

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    # Solve x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
    design = np.column_stack([x, y, np.ones(len(points))])
    rhs = x * x + y * y
    coeffs, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
    center_x = coeffs[0] / 2.0
    center_y = coeffs[1] / 2.0
    radius = float(np.sqrt(max(coeffs[2] + center_x**2 + center_y**2, 0.0)))
    return (float(center_x), float(center_y)), radius


def fit_circle_fixed_center(points: np.ndarray, center: Sequence[float]) -> float:
    """Radius of the best-fit circle through points about a *fixed* centre.

    Used for the outer ring, whose centre is known to coincide with the inner
    ring's. Fixing the centre means a usable radius comes out of even one mark,
    which is what makes a partial outer ring workable.
    """
    if len(points) == 0:
        raise ValueError("at least one point is required")
    offsets = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return float(np.linalg.norm(offsets, axis=1).mean())


def predict_outer_marks(
    inner_marks: np.ndarray,
    center: Sequence[float],
    radius_ratio: float,
) -> np.ndarray:
    """Where the outer marks should be, given the inner ring and the radius ratio.

    The two rings are concentric and share the same angular layout, so each outer
    mark lies along the same ray as its inner counterpart, scaled by the ratio of
    the physical radii.
    """
    origin = np.asarray(center, dtype=np.float64)
    return origin + (np.asarray(inner_marks, dtype=np.float64) - origin) * radius_ratio


def search_mark_near(
    response: np.ndarray,
    target: Sequence[float],
    search_px: int,
    threshold: float,
) -> Optional[Tuple[float, float, float]]:
    """Best crossness peak within ``search_px`` of ``target``.

    Returns (x, y, score) or None if nothing there clears ``threshold``.
    """
    height, width = response.shape
    tx, ty = int(round(target[0])), int(round(target[1]))
    x_lo, x_hi = max(0, tx - search_px), min(width, tx + search_px + 1)
    y_lo, y_hi = max(0, ty - search_px), min(height, ty + search_px + 1)
    if x_hi <= x_lo or y_hi <= y_lo:
        return None

    window = response[y_lo:y_hi, x_lo:x_hi]
    index = int(np.argmax(window))
    dy, dx = divmod(index, window.shape[1])
    score = float(window[dy, dx])
    if score < threshold:
        return None
    return float(x_lo + dx), float(y_lo + dy), score


def sort_points_by_angle(points: np.ndarray, center: Tuple[float, float]) -> np.ndarray:
    """Order points counter-clockwise around a centre."""
    centered = points - np.array(center, dtype=np.float64)
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    return points[np.argsort(angles)]


# ----------------------------------------------------------------------- detection


class Detection:
    """Where the fiducial marks are in one frame.

    Carries ``image`` and ``residual`` alongside the marks so downstream code can
    render the frame and zoom on a mark without reloading or re-filtering.
    """

    def __init__(self, image: np.ndarray, residual: np.ndarray, response: np.ndarray) -> None:
        self.image = image
        self.residual = residual
        self.response = response
        self.inner_marks = np.empty((0, 2), dtype=np.float64)
        self.outer_marks = np.empty((0, 2), dtype=np.float64)
        self.inner_circle: Optional[Tuple[Tuple[float, float], float]] = None
        self.outer_circle: Optional[Tuple[Tuple[float, float], float]] = None
        self.center: Optional[Tuple[float, float]] = None
        self.noise = 0.0
        self.inner_snr = 0.0
        # How the outer ring was established: "detected" (4 marks measured),
        # "partial" (1-3 measured, rest predicted), "predicted" (none measured, ring
        # placed purely from the inner ring and the mm radius ratio), or "" (none).
        self.outer_source = ""
        self.outer_detected = 0
        # Per-frame scale: depends on camera distance, so it is NOT reusable between
        # batches. The ring ratio below is the scale-invariant quantity.
        self.mm_per_px = float("nan")
        self.ring_ratio_used = float("nan")
        self.ring_ratio_measured = float("nan")

    @property
    def found(self) -> bool:
        """True once both rings are established (the outer one may be predicted)."""
        return len(self.inner_marks) == 4 and len(self.outer_marks) == 4

    @property
    def outer_measured(self) -> bool:
        """True only when all four outer marks were actually measured."""
        return self.outer_detected == 4

    @property
    def inner_radius(self) -> float:
        return self.inner_circle[1] if self.inner_circle else float("nan")

    @property
    def outer_radius(self) -> float:
        return self.outer_circle[1] if self.outer_circle else float("nan")

    def edge(self, ring: str = "inner", axis: str = "y") -> Tuple[float, float]:
        """Mean coordinate of a ring's two extreme mark pairs along one axis.

        Returns ``(low_pair_mean, high_pair_mean)`` -- top/bottom for ``axis="y"``,
        left/right for ``axis="x"``. Averaging each mark *pair* rather than taking a
        single mark keeps callers stable against slight target rotation.
        """
        marks = self.inner_marks if ring == "inner" else self.outer_marks
        column = 1 if axis == "y" else 0
        values = np.sort(marks[:, column])
        return float(values[:2].mean()), float(values[-2:].mean())


def detect(
    image: np.ndarray,
    arm_length: int = DEFAULT_ARM_LENGTH,
    bar_width: int = DEFAULT_BAR_WIDTH,
    background_sigma: float = DEFAULT_BACKGROUND_SIGMA,
    min_snr: float = DEFAULT_MIN_SNR,
    inner_radius_mm: float = INNER_RADIUS_MM,
    outer_radius_mm: float = OUTER_RADIUS_MM,
    ring_ratio: Optional[float] = None,
    guided_search_px: int = DEFAULT_GUIDED_SEARCH_PX,
    guided_min_snr: float = DEFAULT_GUIDED_MIN_SNR,
) -> Detection:
    """Locate the fiducial pattern in a grayscale frame.

    The inner ring must be measured. The outer ring is then established in the
    strongest way available: an unconstrained symmetric-quad search first, and
    failing that a guided search along the rays through the inner marks, scaled by
    ``outer_radius_mm / inner_radius_mm``. Marks that still cannot be measured fall
    back to their predicted positions, and ``Detection.outer_source`` records which
    of those happened.
    """
    residual = high_pass(image, background_sigma)
    response = crossness_response(residual, arm_length=arm_length, bar_width=bar_width)
    result = Detection(image, residual, response)

    # Robust noise scale of the flattened frame, used as the significance unit.
    result.noise = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)

    points, scores = find_peaks(response, min_separation=2 * arm_length + 1)
    if len(points) == 0:
        return result

    # Only peaks that stand clear of the noise may form the inner ring.
    significant = points[scores >= min_snr * result.noise]
    result.inner_snr = float(scores[3] / result.noise) if len(scores) >= 4 else 0.0
    if len(significant) < 4:
        return result

    inner = find_symmetric_quad(significant, radius_tolerance=0.10, angle_tolerance_deg=12.0)
    if inner is None:
        return result

    inner = np.array([refine_subpixel(residual, p) for p in inner], dtype=np.float64)
    center = inner.mean(axis=0)
    result.inner_marks = sort_points_by_angle(inner, (float(center[0]), float(center[1])))
    result.center = (float(center[0]), float(center[1]))
    result.inner_circle = fit_circle(result.inner_marks)
    inner_radius = result.inner_circle[1]
    # Physical scale straight off the fiducial: the inner ring is a known size.
    result.mm_per_px = float(inner_radius_mm / inner_radius) if inner_radius > 0 else float("nan")

    centre_tuple = (float(center[0]), float(center[1]))
    # The ratio is what the geometry needs, and it is distance-independent.
    radius_ratio = float(ring_ratio) if ring_ratio else outer_radius_mm / inner_radius_mm
    result.ring_ratio_used = radius_ratio

    # First try the unconstrained search: it needs no prior and is what validates on
    # frames where the outer ring is clean.
    outer = find_symmetric_quad(
        points,
        center,
        min_radius=1.3 * inner_radius,
        max_radius=8.0 * inner_radius,
    )
    if outer is not None:
        outer = np.array([refine_subpixel(residual, p) for p in outer], dtype=np.float64)
        result.outer_marks = sort_points_by_angle(outer, centre_tuple)
        result.outer_circle = fit_circle(result.outer_marks)
        result.outer_detected = 4
        result.outer_source = "detected"
        result.ring_ratio_measured = result.outer_circle[1] / inner_radius
        return result

    # Otherwise use the known geometry: the rings are concentric with a fixed radius
    # ratio, so each outer mark is predictable to within a few px. Search a small
    # window around each prediction and keep whatever is measurable.
    predicted = predict_outer_marks(result.inner_marks, center, radius_ratio)
    threshold = guided_min_snr * result.noise
    marks, measured = [], 0
    for target in predicted:
        hit = search_mark_near(response, target, guided_search_px, threshold)
        if hit is None:
            marks.append(target)  # fall back to the predicted position
        else:
            marks.append(np.array(refine_subpixel(residual, hit[:2]), dtype=np.float64))
            measured += 1

    result.outer_marks = sort_points_by_angle(np.array(marks, dtype=np.float64), centre_tuple)
    result.outer_detected = measured
    result.outer_source = "detected" if measured == 4 else ("partial" if measured else "predicted")
    # Centre is fixed to the inner ring's, so even one measured mark gives a radius.
    if measured:
        measured_marks = np.array(
            [m for m, p in zip(marks, predicted) if not np.allclose(m, p)], dtype=np.float64
        ).reshape(-1, 2)
        radius = fit_circle_fixed_center(measured_marks, center)
    else:
        radius = inner_radius * radius_ratio
    result.outer_circle = (centre_tuple, float(radius))
    if measured:
        result.ring_ratio_measured = radius / inner_radius
    return result


def detect_file(image_path: Path, **kwargs) -> Detection:
    """Load a DNG and detect its fiducial marks."""
    return detect(load_image(image_path), **kwargs)


# -------------------------------------------------------------------- reporting


def detection_to_dict(result: Detection, image_path: Path) -> Dict[str, object]:
    """Serialisable summary of a detection."""

    def circle_dict(circle) -> Optional[dict]:
        if circle is None:
            return None
        (cx, cy), radius = circle
        return {"center": [cx, cy], "radius": radius}

    return {
        "image": image_path.name,
        "pattern_found": result.found,
        "residual_noise": result.noise,
        "inner_mark_snr": result.inner_snr,
        "pattern_center": list(result.center) if result.center else None,
        "inner_marks": [[float(p[0]), float(p[1])] for p in result.inner_marks],
        "outer_marks": [[float(p[0]), float(p[1])] for p in result.outer_marks],
        "inner_circle": circle_dict(result.inner_circle),
        "outer_circle": circle_dict(result.outer_circle),
        "outer_ring_source": result.outer_source,
        "outer_marks_measured": result.outer_detected,
        "mm_per_px": result.mm_per_px,
        "mm_per_px_note": "per-frame; depends on camera distance, not reusable across batches",
        "ring_ratio_used": result.ring_ratio_used,
        "ring_ratio_measured": result.ring_ratio_measured,
        "ring_ratio_note": "scale-invariant; expected constant across batches",
        "ring_radii_mm": {"inner": INNER_RADIUS_MM, "outer": OUTER_RADIUS_MM},
    }


def describe(result: Detection, name: str) -> str:
    """One- or three-line human-readable detection summary."""
    if not result.found:
        if len(result.inner_marks) == 4:
            return f"{name}: inner ring only, no concentric outer ring"
        return f"{name}: no fiducial pattern (mark SNR {result.inner_snr:.2f})"
    outer_note = {
        "detected": "all 4 outer marks measured",
        "partial": f"{result.outer_detected}/4 outer marks measured, rest predicted",
        "predicted": "outer ring predicted from the inner ring and the mm radius ratio",
    }.get(result.outer_source, result.outer_source)
    return (
        f"{name}: inner ring + outer ring, SNR {result.inner_snr:.1f}\n"
        f"  centre = ({result.center[0]:.1f}, {result.center[1]:.1f})\n"
        f"  r_inner = {result.inner_radius:.1f} px, r_outer = {result.outer_radius:.1f} px"
        f"   ({result.mm_per_px * 1000:.2f} um/px)\n"
        f"  outer ring: {outer_note}\n"
        f"  ring ratio: used {result.ring_ratio_used:.4f}"
        + (
            f", measured {result.ring_ratio_measured:.4f}"
            if np.isfinite(result.ring_ratio_measured)
            else " (not measurable in this frame)"
        )
    )


def add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the detection tuning flags, shared with process_images.py."""
    parser.add_argument("--arm-length", type=int, default=DEFAULT_ARM_LENGTH, help="Cross arm length in pixels")
    parser.add_argument("--bar-width", type=int, default=DEFAULT_BAR_WIDTH, help="Cross bar thickness in pixels")
    parser.add_argument(
        "--background-sigma",
        type=float,
        default=DEFAULT_BACKGROUND_SIGMA,
        help="Gaussian sigma used to flatten the illumination gradient",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=DEFAULT_MIN_SNR,
        help="Minimum mark response, in units of residual noise, for a valid detection",
    )
    parser.add_argument(
        "--inner-radius-mm",
        type=float,
        default=INNER_RADIUS_MM,
        help="Physical radius of the inner fiducial ring in mm",
    )
    parser.add_argument(
        "--ring-ratio",
        type=float,
        default=None,
        help="Outer/inner radius ratio to use directly, overriding the mm radii. This "
             "ratio is scale-invariant, so it carries across batches at different distances",
    )
    parser.add_argument(
        "--outer-radius-mm",
        type=float,
        default=OUTER_RADIUS_MM,
        help="Physical radius of the outer fiducial ring in mm; its ratio to the inner "
             "radius predicts the outer marks when they cannot be measured",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect fiducial cross marks in DNG frames")
    parser.add_argument("input", type=Path, help="Path to a DNG file or a directory of DNG files")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for the JSON output")
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Filename glob to filter a directory, e.g. '*cam_2*'",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images from a directory")
    add_detection_arguments(parser)
    args = parser.parse_args()

    images = discover_images(args.input.resolve(), args.glob)
    if not images:
        raise FileNotFoundError(f"No DNG files found in {args.input}")
    if args.limit is not None:
        images = images[: args.limit]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        result = detect_file(
            image_path,
            arm_length=args.arm_length,
            bar_width=args.bar_width,
            background_sigma=args.background_sigma,
            min_snr=args.min_snr,
            inner_radius_mm=args.inner_radius_mm,
            outer_radius_mm=args.outer_radius_mm,
            ring_ratio=args.ring_ratio,
        )
        print(describe(result, image_path.name))

        json_path = output_dir / f"{image_path.stem}_marks.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(detection_to_dict(result, image_path), handle, indent=2)
        print(f"  saved {json_path}")


if __name__ == "__main__":
    main()
