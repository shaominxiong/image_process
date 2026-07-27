#!/usr/bin/env python3
"""Define fiducial-referenced ROIs, extract intensity profiles, normalise, and plot.

Mark detection lives in detect_fiducials.py; this module consumes it and does the
measurement:

1. Place two measurement stripes in the gaps between the two fiducial rings, so
   they sample the illumination without cutting through a mark. ``--orientation``
   selects which pair of gaps to use:

     horizontal (top/bottom gaps)        vertical (left/right gaps)

       *           *  <- outer ring        *  |       |  *
       ============     <- top stripe         |  +   + |
           +   +                              |        |
                        <- centre             |  +   + |
           +   +                           *  |       |  *
       ============     <- bottom stripe      ^         ^
       *           *  <- inner ring        left      right

   Each stripe is 1.05 outer-circle diameters along its long axis and 50 px thick
   by default, centred on the pattern centre along that long axis.
2. Collapse each stripe across its short axis to get intensity along its length
   (vs x for horizontal stripes, vs y for vertical), then add the two together.
3. Fit a second-order polynomial to the added profile and take its maximum.
4. Normalise both the added profile and the fit by that maximum.
5. Plot the frame, the stripe placement, the individual profiles, and the fit.

Also converts the profile axis to off-axis angle and object-plane mm using the lens
model (see PROFILE_FOV_DEG / --object-distance-mm); the vertical field of view is
used when profiling along y.

    python process_images.py <file-or-dir.dng> [--output-dir output] [--show]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from detect_fiducials import (
    Detection,
    add_detection_arguments,
    detect,
    detection_to_dict,
    discover_images,
    load_image,
)
from detect_fiducials import describe as describe_detection

# Measurement stripes: long dimension in outer-circle diameters, short one in pixels.
DEFAULT_STRIPE_LENGTH_FACTOR = 1.05
DEFAULT_STRIPE_THICKNESS = 50
ORIENTATIONS = ("horizontal", "vertical")
DEFAULT_ORIENTATION = "horizontal"
# Lens field of view (horizontal, vertical) in degrees, for the angle/mm axes.
PROFILE_FOV_DEG = (5.6, 4.2)
DEFAULT_OBJECT_DISTANCE_MM = 450.0

RING_STYLE = {"inner": "#e04b4b", "outer": "#3ba7e0"}
STRIPE_COLORS = ("#ffd400", "#7ef542")
# Which gap each stripe sits in, named per orientation.
STRIPE_LABELS = {
    "horizontal": ("top stripe", "bottom stripe"),
    "vertical": ("left stripe", "right stripe"),
}
# The stripe's long axis: profiles run along it.
PROFILE_AXIS = {"horizontal": "x", "vertical": "y"}


# --------------------------------------------------------------------- ROI layout


def _clip_roi(x0: float, y0: float, x1: float, y1: float, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Round an ROI to integer pixels and clip it to the frame."""
    height, width = shape
    return (
        max(0, min(width - 1, int(round(x0)))),
        max(0, min(height - 1, int(round(y0)))),
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
    )


def stripe_rois(
    detection: Detection,
    orientation: str = DEFAULT_ORIENTATION,
    length_factor: float = DEFAULT_STRIPE_LENGTH_FACTOR,
    thickness: int = DEFAULT_STRIPE_THICKNESS,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Two stripes sitting in the gaps between the inner and outer rings.

    Horizontal stripes span x and are stacked in y (top stripe between the
    outer-top and inner-top marks, bottom stripe between the inner-bottom and
    outer-bottom marks). Vertical stripes are the same construction rotated 90
    degrees: they span y and sit in the left and right ring gaps. Either way each
    stripe is centred in its gap and on the pattern centre along its long axis, so
    it never overlaps a mark.

    ``length_factor`` is the long dimension in outer-circle *diameters*;
    ``thickness`` is the short dimension in pixels.
    """
    if not detection.found:
        raise ValueError("both fiducial rings are required to place the stripes")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}")

    # Gaps are measured across the stripes' short axis.
    gap_axis = "y" if orientation == "horizontal" else "x"
    inner_low, inner_high = detection.edge("inner", gap_axis)
    outer_low, outer_high = detection.edge("outer", gap_axis)
    first_pos = (outer_low + inner_low) / 2.0
    second_pos = (inner_high + outer_high) / 2.0

    half_length = length_factor * detection.outer_radius  # length_factor * diameter / 2
    half_thickness = thickness / 2.0
    center_x, center_y = detection.center
    shape = detection.image.shape

    if orientation == "horizontal":
        x0, x1 = center_x - half_length, center_x + half_length
        first = _clip_roi(x0, first_pos - half_thickness, x1, first_pos + half_thickness, shape)
        second = _clip_roi(x0, second_pos - half_thickness, x1, second_pos + half_thickness, shape)
    else:
        y0, y1 = center_y - half_length, center_y + half_length
        first = _clip_roi(first_pos - half_thickness, y0, first_pos + half_thickness, y1, shape)
        second = _clip_roi(second_pos - half_thickness, y0, second_pos + half_thickness, y1, shape)
    return first, second


# ---------------------------------------------------------------- profile extraction


def roi_profile(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    orientation: str = DEFAULT_ORIENTATION,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse an ROI across its short axis to get mean intensity along its length.

    Horizontal stripes average over y and return intensity vs x; vertical stripes
    average over x and return intensity vs y.
    """
    x0, y0, x1, y1 = roi
    patch = image[y0 : y1 + 1, x0 : x1 + 1]
    if orientation == "horizontal":
        return np.arange(x0, x1 + 1, dtype=np.float64), patch.mean(axis=0)
    return np.arange(y0, y1 + 1, dtype=np.float64), patch.mean(axis=1)


def add_profiles(
    image: np.ndarray,
    first_roi: Tuple[int, int, int, int],
    second_roi: Tuple[int, int, int, int],
    orientation: str = DEFAULT_ORIENTATION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the axis coordinate, both stripe profiles, and their sum.

    Both stripes share the same extent along their long axis by construction, so
    they add sample by sample. Summing rather than averaging doubles the signal
    against the per-sample noise; the absolute scale is removed by the
    normalisation later.
    """
    axis_first, first = roi_profile(image, first_roi, orientation)
    axis_second, second = roi_profile(image, second_roi, orientation)
    if len(axis_first) != len(axis_second):
        raise ValueError(
            f"stripe extents differ ({len(axis_first)} vs {len(axis_second)} samples); cannot add profiles"
        )
    return axis_first, first, second, first + second


def fit_second_order(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Least-squares parabola through a profile, plus its peak.

    Fitted via a scaled basis (pixel x values are ~1e3, so a raw Vandermonde fit
    is badly conditioned) and converted back to plain ``c0 + c1*x + c2*x^2``.

    ``peak_*`` is always the largest fitted value inside the measured window, which
    is what the normalisation uses. Whether that is a genuine turning point depends
    on the fit, so three flags record the situation:

    * ``concave``  -- c2 < 0, so the parabola has a maximum at all.
    * ``vertex_x`` -- the analytic turning point (a maximum only if concave).
    * ``peak_is_interior`` -- True only when the reported peak really is an interior
      maximum rather than a value at the edge of the window.
    """
    series = np.polynomial.Polynomial.fit(x, y, 2)
    fitted = series(x)
    c0, c1, c2 = np.polynomial.Polynomial.convert(series).coef

    peak_index = int(np.argmax(fitted))
    vertex_x = float(-c1 / (2.0 * c2)) if c2 != 0 else float("nan")
    concave = bool(c2 < 0)
    vertex_inside = bool(np.isfinite(vertex_x) and x.min() <= vertex_x <= x.max())
    residuals = y - fitted

    return {
        "coefficients": [float(c0), float(c1), float(c2)],
        "fitted": fitted,
        "peak_x": float(x[peak_index]),
        "peak_value": float(fitted[peak_index]),
        "vertex_x": vertex_x,
        "concave": concave,
        "vertex_inside": vertex_inside,
        "peak_is_interior": bool(concave and vertex_inside),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


def normalize_by(values: np.ndarray, reference: float) -> np.ndarray:
    """Scale a curve so that ``reference`` maps to 1.0."""
    if reference == 0:
        raise ValueError("cannot normalise by zero")
    return values / reference


def _focal_px(size: int, fov_deg: float) -> float:
    """Focal length in pixels implied by a sensor extent and its field of view."""
    return (size / 2.0) / math.tan(math.radians(fov_deg / 2.0))


def pixel_to_angle(coord: np.ndarray, size: int, fov_deg: float) -> np.ndarray:
    """Convert pixel coordinates along one axis to off-axis angle in degrees."""
    return np.degrees(np.arctan((coord - size / 2.0) / _focal_px(size, fov_deg)))


def pixel_to_mm(coord: np.ndarray, distance_mm: float, size: int, fov_deg: float) -> np.ndarray:
    """Convert pixel coordinates along one axis to object-plane mm at a distance."""
    return distance_mm * (coord - size / 2.0) / _focal_px(size, fov_deg)


def axis_geometry(image_shape: Tuple[int, int], orientation: str) -> Tuple[int, float]:
    """Sensor extent and field of view for the axis the profile runs along."""
    height, width = image_shape
    if orientation == "horizontal":
        return width, PROFILE_FOV_DEG[0]
    return height, PROFILE_FOV_DEG[1]


# ------------------------------------------------------------------------ analysis


@dataclass
class ProfileResult:
    """Everything measured from one frame's stripes.

    ``first``/``second`` are the two stripes in gap order: top then bottom for
    horizontal stripes, left then right for vertical ones. ``axis`` is the
    coordinate the profiles run along (x for horizontal stripes, y for vertical).
    """

    detection: Detection
    orientation: str
    first_roi: Tuple[int, int, int, int]
    second_roi: Tuple[int, int, int, int]
    axis: np.ndarray
    first_profile: np.ndarray
    second_profile: np.ndarray
    combined: np.ndarray
    fit: Dict[str, object]
    combined_norm: np.ndarray
    fitted_norm: np.ndarray
    axis_angle_deg: np.ndarray
    axis_mm: np.ndarray
    axis_from_center: np.ndarray

    @property
    def peak_value(self) -> float:
        return float(self.fit["peak_value"])

    @property
    def labels(self) -> Tuple[str, str]:
        return STRIPE_LABELS[self.orientation]

    @property
    def axis_name(self) -> str:
        return PROFILE_AXIS[self.orientation]

    @property
    def rois(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        return self.first_roi, self.second_roi


def analyse(
    detection: Detection,
    orientation: str = DEFAULT_ORIENTATION,
    length_factor: float = DEFAULT_STRIPE_LENGTH_FACTOR,
    thickness: int = DEFAULT_STRIPE_THICKNESS,
    object_distance_mm: float = DEFAULT_OBJECT_DISTANCE_MM,
) -> ProfileResult:
    """Place the stripes, extract and add the profiles, fit, and normalise."""
    first_roi, second_roi = stripe_rois(
        detection,
        orientation=orientation,
        length_factor=length_factor,
        thickness=thickness,
    )
    axis, first, second, combined = add_profiles(detection.image, first_roi, second_roi, orientation)

    fit = fit_second_order(axis, combined)
    peak_value = fit["peak_value"]
    size, fov = axis_geometry(detection.image.shape, orientation)
    center_along_axis = detection.center[0] if orientation == "horizontal" else detection.center[1]

    return ProfileResult(
        detection=detection,
        orientation=orientation,
        first_roi=first_roi,
        second_roi=second_roi,
        axis=axis,
        first_profile=first,
        second_profile=second,
        combined=combined,
        fit=fit,
        combined_norm=normalize_by(combined, peak_value),
        fitted_norm=normalize_by(fit["fitted"], peak_value),
        axis_angle_deg=pixel_to_angle(axis, size, fov),
        axis_mm=pixel_to_mm(axis, object_distance_mm, size, fov),
        axis_from_center=axis - center_along_axis,
    )


# ------------------------------------------------------------------------ plotting


def stretch_for_display(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Percentile-stretch an image to uint8 so 16-bit frames are actually visible."""
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _draw_ring(axis, marks: np.ndarray, circle, color: str, label: str) -> None:
    """Draw a ring's marks and its fitted circle."""
    if len(marks) == 0:
        return
    axis.plot(marks[:, 0], marks[:, 1], "+", color=color, markersize=14, markeredgewidth=2.0, label=label)
    if circle is not None:
        (cx, cy), radius = circle
        axis.add_patch(Circle((cx, cy), radius, fill=False, edgecolor=color, linewidth=1.2, linestyle="--", alpha=0.9))


def _draw_marks(axis, detection: Detection) -> None:
    _draw_ring(axis, detection.inner_marks, detection.inner_circle, RING_STYLE["inner"], "inner ring")
    _draw_ring(axis, detection.outer_marks, detection.outer_circle, RING_STYLE["outer"], "outer ring")
    if detection.center is not None:
        axis.plot(
            detection.center[0],
            detection.center[1],
            "x",
            color="#f5c542",
            markersize=12,
            markeredgewidth=2.0,
            label="pattern centre",
        )


def _draw_stripes(axis, result: ProfileResult, linewidth: float = 1.8) -> None:
    """Draw the two measurement stripes."""
    for roi, color, label in zip(result.rois, STRIPE_COLORS, result.labels):
        x0, y0, x1, y1 = roi
        axis.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=linewidth, label=label)
        )


def plot_no_pattern(detection: Detection, image_name: str, output_path: Path):
    """Single-panel figure for frames where the pattern was not found."""
    fig, axis = plt.subplots(figsize=(12, 9))
    axis.imshow(stretch_for_display(detection.image), cmap="gray", vmin=0, vmax=255)
    _draw_marks(axis, detection)
    axis.set_title(f"No fiducial pattern detected - {image_name}  (best mark SNR {detection.inner_snr:.2f})")
    axis.set_xlabel("x (px)")
    axis.set_ylabel("y (px)")
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    return fig


def plot_results(result: ProfileResult, image_name: str, output_path: Path):
    """Four-row figure: frame, stripe placement, stripe profiles, fit and normalisation."""
    detection = result.detection
    display = stretch_for_display(detection.image)
    fig = plt.figure(figsize=(15, 16))
    grid = fig.add_gridspec(4, 6, height_ratios=[2.0, 1.3, 1.0, 1.0], hspace=0.34, wspace=0.55)

    axis_name = result.axis_name

    overview = fig.add_subplot(grid[0, :])
    overview.imshow(display, cmap="gray", vmin=0, vmax=255)
    _draw_marks(overview, detection)
    _draw_stripes(overview, result)
    overview.set_title(
        f"Fiducial detection and {result.orientation} measurement stripes - {image_name}"
        f"  (mark SNR {detection.inner_snr:.1f})"
    )
    overview.legend(loc="upper right", framealpha=0.85, fontsize=9)
    overview.set_xlabel("x (px)")
    overview.set_ylabel("y (px)")

    # Zoom on the pattern to confirm each stripe sits in the ring gap, clear of the marks.
    # Bounds are the union of both stripes and the outer ring, so this works either way round.
    zoom = fig.add_subplot(grid[1, 0:4])
    pad = 60
    height, width = detection.image.shape
    boxes = np.array(result.rois, dtype=np.float64)
    zx0 = max(0, int(min(boxes[:, 0].min(), detection.outer_marks[:, 0].min())) - pad)
    zx1 = min(width - 1, int(max(boxes[:, 2].max(), detection.outer_marks[:, 0].max())) + pad)
    zy0 = max(0, int(min(boxes[:, 1].min(), detection.outer_marks[:, 1].min())) - pad)
    zy1 = min(height - 1, int(max(boxes[:, 3].max(), detection.outer_marks[:, 1].max())) + pad)
    zoom.imshow(display, cmap="gray", vmin=0, vmax=255)
    _draw_marks(zoom, detection)
    _draw_stripes(zoom, result, linewidth=2.2)
    zoom.set_xlim(zx0, zx1)
    zoom.set_ylim(zy1, zy0)
    zoom.set_aspect("equal")
    zoom.set_title("Stripe placement between the rings", fontsize=10)
    zoom.set_xlabel("x (px)")
    zoom.set_ylabel("y (px)")

    # One mark crop, to show the sub-pixel centre landing on the arm intersection.
    crop_half = 30
    crop_axis = fig.add_subplot(grid[1, 4:6])
    cx, cy = detection.inner_marks[0]
    x0, y0 = int(round(cx)) - crop_half, int(round(cy)) - crop_half
    patch = detection.residual[y0 : y0 + 2 * crop_half, x0 : x0 + 2 * crop_half]
    crop_axis.imshow(patch, cmap="magma", extent=(x0, x0 + 2 * crop_half, y0 + 2 * crop_half, y0))
    crop_axis.plot(cx, cy, "+", color="#7ef542", markersize=14, markeredgewidth=2.0)
    crop_axis.set_title(f"inner mark @ ({cx:.1f}, {cy:.1f})", fontsize=9)

    # Each stripe collapsed across its short axis -> intensity along its length.
    across = "y" if result.orientation == "horizontal" else "x"
    span = (1, 3) if result.orientation == "horizontal" else (0, 2)
    profile = fig.add_subplot(grid[2, :])
    for values, roi, color, label in zip(
        (result.first_profile, result.second_profile),
        result.rois,
        STRIPE_COLORS,
        result.labels,
    ):
        profile.plot(
            result.axis,
            values,
            color=color,
            linewidth=1.4,
            label=f"{label}  {across} {roi[span[0]]}-{roi[span[1]]}",
        )
    profile.set_xlabel(f"{axis_name} (px)")
    profile.set_ylabel("mean intensity")
    profile.set_title(f"Stripe intensity profile along {axis_name} (mean across {across})", fontsize=10)
    profile.legend(fontsize=8)
    profile.grid(alpha=0.25)

    # Added profile with its second-order fit, then both normalised by the fitted peak.
    fit = result.fit
    fit_axis = fig.add_subplot(grid[3, 0:3])
    fit_axis.plot(result.axis, result.combined, color="#8a8f98", linewidth=1.2, label=" + ".join(result.labels))
    fit_axis.plot(result.axis, fit["fitted"], color="#d7263d", linewidth=2.0, label="2nd-order fit")
    fit_axis.plot(
        fit["peak_x"],
        result.peak_value,
        "o",
        color="#d7263d",
        markersize=7,
        label=f"fitted max = {result.peak_value:.0f}",
    )
    fit_axis.set_xlabel(f"{axis_name} (px)")
    fit_axis.set_ylabel("summed intensity")
    fit_axis.set_title(f"Added profile and 2nd-order fit (RMSE {fit['rmse']:.0f})", fontsize=10)
    fit_axis.legend(fontsize=8)
    fit_axis.grid(alpha=0.25)

    norm_axis = fig.add_subplot(grid[3, 3:6])
    norm_axis.plot(result.axis, result.combined_norm, color="#8a8f98", linewidth=1.2, label="added profile")
    norm_axis.plot(result.axis, result.fitted_norm, color="#d7263d", linewidth=2.0, label="2nd-order fit")
    norm_axis.axhline(1.0, color="#3ba7e0", linewidth=1.0, linestyle="--", alpha=0.8)
    norm_axis.set_xlabel(f"{axis_name} (px)")
    norm_axis.set_ylabel("normalised intensity")
    norm_axis.set_title("Normalised by the fitted maximum", fontsize=10)
    norm_axis.legend(fontsize=8)
    norm_axis.grid(alpha=0.25)

    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    return fig


def save_overlay(result: Optional[ProfileResult], detection: Detection, output_path: Path) -> None:
    """Save a standalone annotated overlay at native resolution."""
    display = stretch_for_display(detection.image)
    height, width = display.shape
    fig = plt.figure(figsize=(width / 200.0, height / 200.0), dpi=200)
    axis = fig.add_axes([0, 0, 1, 1])
    axis.imshow(display, cmap="gray", vmin=0, vmax=255)
    axis.set_axis_off()
    _draw_marks(axis, detection)
    if result is not None:
        _draw_stripes(axis, result, linewidth=2.5)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------- output


def save_profile_csv(path: Path, result: ProfileResult) -> None:
    """Write the per-sample profiles, the fit, and both normalised curves.

    Column names carry the profile axis (x for horizontal stripes, y for vertical)
    and the two stripe labels, so the file is self-describing either way round.
    """
    axis_name = result.axis_name
    first_label, second_label = (label.split()[0] for label in result.labels)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                f"{axis_name}_pixels",
                f"{axis_name}_from_center_px",
                f"{axis_name}_angle_deg",
                f"{axis_name}_mm",
                first_label,
                second_label,
                "combined",
                "fitted",
                "combined_norm",
                "fitted_norm",
            ]
        )
        writer.writerows(
            zip(
                result.axis,
                result.axis_from_center,
                result.axis_angle_deg,
                result.axis_mm,
                result.first_profile,
                result.second_profile,
                result.combined,
                result.fit["fitted"],
                result.combined_norm,
                result.fitted_norm,
            )
        )


def result_to_dict(result: ProfileResult, detection_summary: Dict[str, object]) -> Dict[str, object]:
    """Merge the detection summary with the stripe ROIs and fit parameters."""
    fit = result.fit
    first_key, second_key = (label.replace(" ", "_") + "_roi" for label in result.labels)
    return {
        **detection_summary,
        "stripe_orientation": result.orientation,
        "profile_axis": result.axis_name,
        first_key: list(result.first_roi),
        second_key: list(result.second_roi),
        "roi_format": "x0, y0, x1, y1 (inclusive pixel bounds)",
        "combined_profile_fit": {
            "model": f"combined = c0 + c1*{result.axis_name} + c2*{result.axis_name}^2",
            "coefficients": fit["coefficients"],
            "rmse": fit["rmse"],
            "peak_x": fit["peak_x"],
            "peak_value": fit["peak_value"],
            "vertex_x": fit["vertex_x"],
            "concave": fit["concave"],
            "vertex_inside_stripe": fit["vertex_inside"],
            "peak_is_interior": fit["peak_is_interior"],
            "normalized_by": fit["peak_value"],
        },
    }


def describe(result: ProfileResult) -> str:
    """Human-readable measurement summary."""
    fit = result.fit
    axis_name = result.axis_name
    c0, c1, c2 = fit["coefficients"]
    width = max(len(label) for label in result.labels)
    lines = []
    for roi, label in zip(result.rois, result.labels):
        lines.append(
            f"  {label:<{width}} (x0,y0,x1,y1) = {roi}   "
            f"{roi[2] - roi[0] + 1} x {roi[3] - roi[1] + 1} px"
        )
    lines += [
        f"  added profile fit: {c0:.5g} + {c1:.5g}*{axis_name} + {c2:.5g}*{axis_name}^2   RMSE {fit['rmse']:.0f}",
        f"  fitted max = {fit['peak_value']:.1f} at {axis_name} = {fit['peak_x']:.1f}  (normalisation reference)",
    ]
    if not fit["peak_is_interior"]:
        window = f"{result.axis.min():.0f}-{result.axis.max():.0f}"
        if not fit["concave"]:
            reason = (
                f"fit is convex (c2 = {c2:+.4g}), so it has a minimum, not a maximum; "
                f"the profile is monotonic across the stripe"
            )
        else:
            reason = f"parabola vertex at {axis_name} = {fit['vertex_x']:.0f} lies outside the stripe ({window})"
        lines.append(f"  NOTE: {reason}; the fitted max is a window edge, not a true peak")
    return "\n".join(lines)


# ------------------------------------------------------------------------- driver


def process_image(
    image_path: Path,
    output_dir: Path,
    detection_kwargs: Optional[Dict[str, object]] = None,
    orientation: str = DEFAULT_ORIENTATION,
    stripe_length_factor: float = DEFAULT_STRIPE_LENGTH_FACTOR,
    stripe_thickness: int = DEFAULT_STRIPE_THICKNESS,
    object_distance_mm: float = DEFAULT_OBJECT_DISTANCE_MM,
    show: bool = False,
) -> Optional[Path]:
    """Detect, measure, and plot one frame. Returns the plot path."""
    image = load_image(image_path)
    detection = detect(image, **(detection_kwargs or {}))
    print(describe_detection(detection, image_path.name))

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{image_path.stem}_profile.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    json_path = output_dir / f"{image_path.stem}_profile.json"
    csv_path = output_dir / f"{image_path.stem}_profile.csv"

    summary = detection_to_dict(detection, image_path)

    if not detection.found:
        fig = plot_no_pattern(detection, image_path.name, plot_path)
        save_overlay(None, detection, overlay_path)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        if show:
            plt.show()
        plt.close(fig)
        print(f"  saved {plot_path} (no stripes placed)")
        return plot_path

    result = analyse(
        detection,
        orientation=orientation,
        length_factor=stripe_length_factor,
        thickness=stripe_thickness,
        object_distance_mm=object_distance_mm,
    )
    print(describe(result))

    fig = plot_results(result, image_path.name, plot_path)
    save_overlay(result, detection, overlay_path)
    save_profile_csv(csv_path, result)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result_to_dict(result, summary), handle, indent=2)

    if show:
        plt.show()
    plt.close(fig)

    print(f"  saved {plot_path}")
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, normalise, and plot fiducial-referenced intensity profiles from DNG frames"
    )
    parser.add_argument("input", type=Path, help="Path to a DNG file or a directory of DNG files")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for outputs")
    parser.add_argument(
        "--orientation",
        choices=ORIENTATIONS,
        default=DEFAULT_ORIENTATION,
        help=(
            "Stripe orientation: 'horizontal' puts them in the top/bottom ring gaps "
            "and profiles along x; 'vertical' puts them in the left/right gaps and profiles along y"
        ),
    )
    parser.add_argument(
        "--stripe-length-factor",
        "--stripe-width-factor",
        dest="stripe_length_factor",
        type=float,
        default=DEFAULT_STRIPE_LENGTH_FACTOR,
        help="Stripe long dimension in outer-circle diameters",
    )
    parser.add_argument(
        "--stripe-thickness",
        "--stripe-height",
        dest="stripe_thickness",
        type=int,
        default=DEFAULT_STRIPE_THICKNESS,
        help="Stripe short dimension in pixels",
    )
    parser.add_argument(
        "--object-distance-mm",
        type=float,
        default=DEFAULT_OBJECT_DISTANCE_MM,
        help="Object distance in mm used for the mm-axis conversion",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure after processing")
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Filename glob to filter a directory, e.g. '*cam_2*'",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images from a directory")
    add_detection_arguments(parser)
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)

    images = discover_images(args.input.resolve(), args.glob)
    if not images:
        raise FileNotFoundError(f"No DNG files found in {args.input}")
    print(f"Processing {len(images)} image(s) -> {args.output_dir.resolve()}\n")
    if args.limit is not None:
        images = images[: args.limit]

    detection_kwargs = {
        "arm_length": args.arm_length,
        "bar_width": args.bar_width,
        "background_sigma": args.background_sigma,
        "min_snr": args.min_snr,
    }

    for image_path in images:
        process_image(
            image_path,
            args.output_dir.resolve(),
            detection_kwargs=detection_kwargs,
            orientation=args.orientation,
            stripe_length_factor=args.stripe_length_factor,
            stripe_thickness=args.stripe_thickness,
            object_distance_mm=args.object_distance_mm,
            show=args.show,
        )


if __name__ == "__main__":
    main()
