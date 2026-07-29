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
   (vs x for horizontal stripes, vs y for vertical). The two stripes are kept
   separate -- they are never summed.
3. Fit a second-order polynomial to each stripe's profile independently.
4. Normalise every curve by one shared reference: the larger of the two fitted
   maxima. The brighter stripe then peaks at 1.0 and the relative offset between
   the two stripes is preserved, so they stay directly comparable.
5. Plot the frame, the stripe placement, both profiles, and both fits.

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
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from detect_fiducials import (
    CHANNELS,
    DEFAULT_CHANNEL,
    Detection,
    add_detection_arguments,
    add_loading_arguments,
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
# "auto" is accepted at the CLI: it measures the gradient across the outer circle and
# runs the stripes along it, which needs the rotated sampling path.
ORIENTATION_CHOICES = ("horizontal", "vertical", "auto")
DEFAULT_ORIENTATION = "horizontal"
# Where each stripe sits relative to the ring gap it belongs to.
STRIPE_POSITIONS = ("centred", "outer", "beyond")
DEFAULT_STRIPE_POSITION = "outer"
# Which ring the stripes are positioned against. "inner" allows frames where the
# outer ring is missing or too faint to detect; scale then comes from r_inner.
STRIPE_REFERENCES = ("outer", "inner")
DEFAULT_STRIPE_REFERENCE = "outer"
# Clearance in px between a stripe edge and the nearest mark's measured arm tip.
DEFAULT_MARK_CLEARANCE = 20.0
# Extra outward shift in px applied to both stripes, away from the pattern centre.
DEFAULT_STRIPE_OFFSET = 0.0
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
# Stripe angle in degrees for the two axis-aligned orientations, measuring the long
# axis anticlockwise from +x. Any other angle uses the rotated sampling path.
ORIENTATION_ANGLE = {"horizontal": 0.0, "vertical": 90.0}
# Gradient measurement: smoothing applied before differencing, and the magnitude
# (counts/px) below which the direction is treated as undetermined -- near the beam
# centre the gradient vanishes and its direction is pure noise.
DEFAULT_GRADIENT_SIGMA = 15.0
DEFAULT_MIN_GRADIENT = 0.30


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


def measure_gradient_direction(
    detection: Detection,
    sigma: float = DEFAULT_GRADIENT_SIGMA,
    radius_scale: float = 1.0,
    min_gradient: float = DEFAULT_MIN_GRADIENT,
) -> Dict[str, float]:
    """Direction of steepest intensity change across the outer circle.

    The illumination is a smooth ramp over this region, so the *net* gradient vector
    averaged over the disc gives the falloff direction. Marks are small and
    symmetric about the centre, so they contribute almost nothing to that mean.

    Returns the angle of the gradient in degrees (0 = +x, 90 = +y, i.e. downward in
    image coordinates), its magnitude in counts/px, and ``determined``: False when
    the magnitude is below ``min_gradient``, which happens near the beam centre
    where the gradient genuinely vanishes and its direction is meaningless.
    """
    image = detection.image
    smooth = gaussian_filter(image, sigma)
    grad_y, grad_x = np.gradient(smooth)

    cx, cy = detection.center
    radius = detection.outer_radius * radius_scale
    height, width = image.shape
    y0, y1 = max(0, int(cy - radius)), min(height, int(cy + radius) + 1)
    x0, x1 = max(0, int(cx - radius)), min(width, int(cx + radius) + 1)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    mean_gx = float(grad_x[y0:y1, x0:x1][inside].mean())
    mean_gy = float(grad_y[y0:y1, x0:x1][inside].mean())

    magnitude = float(np.hypot(mean_gx, mean_gy))
    angle = float(np.degrees(np.arctan2(mean_gy, mean_gx)))
    return {
        "angle_deg": angle % 180.0,  # a stripe direction is mod 180
        "signed_angle_deg": angle,
        "magnitude": magnitude,
        "gx": mean_gx,
        "gy": mean_gy,
        "determined": bool(magnitude >= min_gradient),
        "nearest_axis": "horizontal" if abs(mean_gx) >= abs(mean_gy) else "vertical",
    }


def resolve_stripe_angle(
    detection: Detection,
    orientation: str,
    stripe_angle: Optional[float] = None,
    fallback_angle: float = 0.0,
    min_gradient: float = DEFAULT_MIN_GRADIENT,
    gradient_sigma: float = DEFAULT_GRADIENT_SIGMA,
) -> Dict[str, object]:
    """Decide the stripe long-axis angle, and say where it came from.

    ``stripe_angle`` wins if given. ``orientation="auto"`` measures the gradient
    across the outer circle and uses that direction, falling back to
    ``fallback_angle`` when the gradient is too weak to define one.
    """
    if stripe_angle is not None:
        return {"angle": float(stripe_angle) % 180.0, "source": "fixed", "gradient": None}

    if orientation != "auto":
        return {"angle": ORIENTATION_ANGLE[orientation], "source": orientation, "gradient": None}

    gradient = measure_gradient_direction(
        detection, sigma=gradient_sigma, min_gradient=min_gradient
    )
    if gradient["determined"]:
        return {"angle": gradient["angle_deg"], "source": "gradient", "gradient": gradient}
    return {"angle": float(fallback_angle) % 180.0, "source": "fallback", "gradient": gradient}


def _unit_vectors(angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Along-stripe and across-stripe unit vectors for a long-axis angle."""
    radians = np.radians(angle_deg)
    along = np.array([np.cos(radians), np.sin(radians)], dtype=np.float64)
    across = np.array([-np.sin(radians), np.cos(radians)], dtype=np.float64)
    return along, across


def measure_mark_extent(
    detection: Detection,
    ring: str,
    axis: str,
    threshold_sigma: float = 2.0,
    search: int = 60,
) -> float:
    """Half-extent of a ring's mark arms along one axis, in pixels.

    Measured from the high-passed residual rather than assumed, because the arms
    are much longer than the detector's template (~30 px, not the 20 px arm box)
    and the outer marks are larger than the inner ones. Returns the largest
    half-extent over the ring's four marks, so clearances computed from it are
    conservative.
    """
    marks = detection.inner_marks if ring == "inner" else detection.outer_marks
    residual = detection.residual
    threshold = threshold_sigma * detection.noise
    height, width = residual.shape

    extents = [0.0]
    for mx, my in marks:
        cx, cy = int(round(mx)), int(round(my))
        if axis == "x":
            lo, hi = max(0, cx - search), min(width, cx + search + 1)
            profile = residual[max(0, cy - 1) : cy + 2, lo:hi].mean(axis=0)
            offset = cx - lo
        else:
            lo, hi = max(0, cy - search), min(height, cy + search + 1)
            profile = residual[lo:hi, max(0, cx - 1) : cx + 2].mean(axis=1)
            offset = cy - lo
        above = np.nonzero(profile > threshold)[0]
        if len(above):
            extents.append(float(max(offset - above.min(), above.max() - offset)))
    return max(extents)


def stripe_rois(
    detection: Detection,
    orientation: str = DEFAULT_ORIENTATION,
    length_factor: float = DEFAULT_STRIPE_LENGTH_FACTOR,
    thickness: int = DEFAULT_STRIPE_THICKNESS,
    position: str = DEFAULT_STRIPE_POSITION,
    mark_clearance: float = DEFAULT_MARK_CLEARANCE,
    offset: float = DEFAULT_STRIPE_OFFSET,
    reference: str = DEFAULT_STRIPE_REFERENCE,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Two stripes placed relative to the inner/outer ring gaps.

    Horizontal stripes span x and are stacked in y (first stripe on the top side,
    second on the bottom). Vertical stripes are the same construction rotated 90
    degrees: they span y and sit on the left and right sides.

    ``position`` sets where each stripe sits relative to the reference ring. All
    keep ``mark_clearance`` px between the stripe and the nearest mark's *measured*
    arm tip, so none of them overlaps a mark:

    * ``centred`` -- midway between the inner and outer mark pairs (needs both rings).
    * ``outer``   -- pushed inward from the reference marks, up against them.
    * ``beyond``  -- outside the reference ring entirely, just past its marks.

    ``reference`` chooses which ring the stripes are placed against, and which
    radius sets the stripe length. ``"inner"`` lets frames with a missing or too
    faint outer ring still be measured -- but the stripes then sit at a different
    distance from the pattern centre, so such results are NOT comparable with
    outer-referenced ones.

    ``offset`` then nudges both stripes further outward (away from the pattern
    centre) by that many pixels, for fine positioning on top of any mode.

    ``length_factor`` is the long dimension in reference-circle *diameters*;
    ``thickness`` is the short dimension in pixels.
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}")
    if position not in STRIPE_POSITIONS:
        raise ValueError(f"position must be one of {STRIPE_POSITIONS}, got {position!r}")
    if reference not in STRIPE_REFERENCES:
        raise ValueError(f"reference must be one of {STRIPE_REFERENCES}, got {reference!r}")
    if len(detection.inner_marks) != 4:
        raise ValueError("the inner fiducial ring is required to place the stripes")
    if reference == "outer" and len(detection.outer_marks) != 4:
        raise ValueError("outer-referenced stripes need the outer ring; retry with reference='inner'")
    if position == "centred" and len(detection.outer_marks) != 4:
        raise ValueError("the 'centred' position needs both rings")

    # Gaps are measured across the stripes' short axis.
    gap_axis = "y" if orientation == "horizontal" else "x"
    half_thickness = thickness / 2.0

    if position == "centred":
        inner_low, inner_high = detection.edge("inner", gap_axis)
        outer_low, outer_high = detection.edge("outer", gap_axis)
        first_pos = (outer_low + inner_low) / 2.0
        second_pos = (inner_high + outer_high) / 2.0
    else:
        ref_low, ref_high = detection.edge(reference, gap_axis)
        ref_extent = measure_mark_extent(detection, reference, gap_axis)
        # Distance from a reference mark centre to the stripe centre, so the near
        # edge clears that mark's arm tip by mark_clearance.
        offset_to_edge = ref_extent + mark_clearance + half_thickness
        if position == "outer":
            # Inward from the reference marks, hugging them.
            first_pos = ref_low + offset_to_edge
            second_pos = ref_high - offset_to_edge
        else:  # beyond: outside the reference ring
            first_pos = ref_low - offset_to_edge
            second_pos = ref_high + offset_to_edge

    # Extra outward nudge, away from the pattern centre on both sides.
    first_pos -= offset
    second_pos += offset

    ref_radius = detection.inner_radius if reference == "inner" else detection.outer_radius
    half_length = length_factor * ref_radius  # length_factor * diameter / 2
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


@dataclass
class Stripe:
    """A stripe of arbitrary orientation, used when the angle is not 0 or 90 deg."""

    centre: np.ndarray  # (x, y)
    angle: float  # long-axis angle in degrees, anticlockwise from +x
    half_length: float
    half_thickness: float

    @property
    def along(self) -> np.ndarray:
        return _unit_vectors(self.angle)[0]

    @property
    def across(self) -> np.ndarray:
        return _unit_vectors(self.angle)[1]

    def corners(self) -> np.ndarray:
        """Four corners in order, for drawing."""
        along, across = _unit_vectors(self.angle)
        return np.array(
            [
                self.centre + along * self.half_length + across * self.half_thickness,
                self.centre + along * self.half_length - across * self.half_thickness,
                self.centre - along * self.half_length - across * self.half_thickness,
                self.centre - along * self.half_length + across * self.half_thickness,
            ]
        )

    def bounding_box(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Axis-aligned box enclosing the stripe, for reporting alongside the boxes."""
        c = self.corners()
        return _clip_roi(c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max(), shape)


def sample_stripe(image: np.ndarray, stripe: Stripe, step: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Mean intensity along a rotated stripe, sampled by bilinear interpolation.

    Returns the signed distance along the stripe from its centre (px) and the profile
    averaged across its thickness.
    """
    along, across = _unit_vectors(stripe.angle)
    s = np.arange(-stripe.half_length, stripe.half_length + step / 2.0, step)
    t = np.arange(-stripe.half_thickness, stripe.half_thickness + 0.5, 1.0)

    # Grid of shape (across samples, along samples), then average across.
    xs = stripe.centre[0] + along[0] * s[None, :] + across[0] * t[:, None]
    ys = stripe.centre[1] + along[1] * s[None, :] + across[1] * t[:, None]

    values = map_coordinates(image, [ys, xs], order=1, mode="nearest")
    return s, values.mean(axis=0)


def measure_mark_extent_along(
    detection: Detection,
    ring: str,
    direction: np.ndarray,
    threshold_sigma: float = 2.0,
    search: int = 60,
) -> float:
    """Half-extent of a ring's mark arms along an arbitrary direction, in px.

    The rotated analogue of measure_mark_extent: samples the residual along
    ``direction`` through each mark and takes the largest run above threshold.
    """
    marks = detection.inner_marks if ring == "inner" else detection.outer_marks
    residual = detection.residual
    threshold = threshold_sigma * detection.noise
    offsets = np.arange(-search, search + 1, 1.0)

    extents = [0.0]
    for mark in marks:
        xs = mark[0] + direction[0] * offsets
        ys = mark[1] + direction[1] * offsets
        profile = map_coordinates(residual, [ys, xs], order=1, mode="nearest")
        above = np.nonzero(profile > threshold)[0]
        if len(above):
            centre_index = len(offsets) // 2
            extents.append(float(max(centre_index - above.min(), above.max() - centre_index)))
    return max(extents)


def rotated_stripes(
    detection: Detection,
    angle: float,
    length_factor: float = DEFAULT_STRIPE_LENGTH_FACTOR,
    thickness: int = DEFAULT_STRIPE_THICKNESS,
    position: str = DEFAULT_STRIPE_POSITION,
    mark_clearance: float = DEFAULT_MARK_CLEARANCE,
    offset: float = DEFAULT_STRIPE_OFFSET,
    reference: str = DEFAULT_STRIPE_REFERENCE,
) -> Tuple[Stripe, Stripe]:
    """Two stripes at ``angle``, offset either side of the centre along its normal.

    Same construction as the axis-aligned case with the gap axis replaced by the
    stripe normal, so gaps and clearances are measured perpendicular to the stripes.
    """
    if len(detection.inner_marks) != 4:
        raise ValueError("the inner fiducial ring is required to place the stripes")

    centre = np.array(detection.center, dtype=np.float64)
    across = _unit_vectors(angle)[1]
    half_thickness = thickness / 2.0

    def projections(ring: str) -> np.ndarray:
        """Signed distances of a ring's marks from the centre along the normal."""
        marks = detection.inner_marks if ring == "inner" else detection.outer_marks
        return (marks - centre) @ across

    # A tilted stripe can clip a mark that sits off to one side even when the pair
    # *means* look clear, so bound against the individual extreme marks. Treat each
    # mark as a disc of its arm extent, which contains the cross.
    mark_radius = max(
        measure_mark_extent_along(detection, ring, across)
        for ring in ("inner", "outer")
        if len(detection.inner_marks if ring == "inner" else detection.outer_marks) == 4
    )

    if position == "centred":
        inner = projections("inner")
        outer = projections("outer")
        first_d = (outer.min() + inner.min()) / 2.0
        second_d = (inner.max() + outer.max()) / 2.0
    else:
        ref = projections(reference)
        step = mark_radius + mark_clearance + half_thickness
        if position == "outer":
            first_d, second_d = ref.min() + step, ref.max() - step
        else:  # beyond: clear of every mark on that side, whatever its position along s
            all_proj = np.concatenate(
                [projections(r) for r in ("inner", "outer")
                 if len((detection.inner_marks if r == "inner" else detection.outer_marks)) == 4]
            )
            first_d = all_proj.min() - step
            second_d = all_proj.max() + step

    first_d -= offset
    second_d += offset

    ref_radius = detection.inner_radius if reference == "inner" else detection.outer_radius
    half_length = length_factor * ref_radius
    return (
        Stripe(centre + across * first_d, angle, half_length, half_thickness),
        Stripe(centre + across * second_d, angle, half_length, half_thickness),
    )


def rotated_clearances(detection: Detection, stripes: Tuple[Stripe, Stripe], labels: Tuple[str, str]) -> Dict[str, float]:
    """True 2D gap in px between each rotated stripe rectangle and the nearest mark.

    Each mark is transformed into the stripe's own (along, across) frame and its
    distance to the rectangle taken there. Projecting marks onto the normal alone --
    the axis-aligned shortcut -- is wrong once the stripe is tilted, because it
    discards where the mark sits along the stripe and so misses real overlaps.
    """
    marks = [m for ring in ("inner", "outer")
             for m in (detection.inner_marks if ring == "inner" else detection.outer_marks)]
    if not marks:
        return {label: float("nan") for label in labels}

    across = stripes[0].across
    radius = max(
        measure_mark_extent_along(detection, ring, across)
        for ring in ("inner", "outer")
        if len((detection.inner_marks if ring == "inner" else detection.outer_marks)) == 4
    )

    clearances = {}
    for stripe, label in zip(stripes, labels):
        along, normal = _unit_vectors(stripe.angle)
        gaps = []
        for mark in marks:
            delta = np.asarray(mark, dtype=np.float64) - stripe.centre
            s, t = float(delta @ along), float(delta @ normal)
            ds = max(0.0, abs(s) - stripe.half_length)
            dt = max(0.0, abs(t) - stripe.half_thickness)
            gaps.append(float(np.hypot(ds, dt)) - radius)
        clearances[label] = min(gaps)
    return clearances


def stripe_clearances(
    detection: Detection,
    rois: Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]],
    orientation: str,
) -> Dict[str, float]:
    """Gap in px between each stripe and the nearest mark arm tip.

    Positive means clear, negative means the stripe cuts into a mark. Measured
    against the real arm extents, so this is the check that "close but no overlap"
    actually holds.
    """
    gap_axis = "y" if orientation == "horizontal" else "x"
    lo_index, hi_index = (1, 3) if orientation == "horizontal" else (0, 2)

    bands = []  # (low, high) span occupied by each mark pair along the gap axis
    for ring in ("inner", "outer"):
        marks = detection.inner_marks if ring == "inner" else detection.outer_marks
        if len(marks) != 4:
            continue  # ring absent in this frame
        extent = measure_mark_extent(detection, ring, gap_axis)
        low, high = detection.edge(ring, gap_axis)
        bands.append((low - extent, low + extent))
        bands.append((high - extent, high + extent))

    clearances = {}
    for roi, label in zip(rois, STRIPE_LABELS[orientation]):
        stripe = (float(roi[lo_index]), float(roi[hi_index]))
        # Positive gap if the intervals are disjoint; negative if they overlap.
        gaps = [max(band[0] - stripe[1], stripe[0] - band[1]) for band in bands]
        clearances[label] = float(min(gaps))
    return clearances


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


def stripe_profiles(
    image: np.ndarray,
    first_roi: Tuple[int, int, int, int],
    second_roi: Tuple[int, int, int, int],
    orientation: str = DEFAULT_ORIENTATION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the shared axis coordinate and each stripe's profile, kept separate.

    Both stripes span the same range along their long axis by construction, so a
    single axis array describes both and they stay sample-aligned for comparison.
    """
    axis_first, first = roi_profile(image, first_roi, orientation)
    axis_second, second = roi_profile(image, second_roi, orientation)
    if len(axis_first) != len(axis_second):
        raise ValueError(
            f"stripe extents differ ({len(axis_first)} vs {len(axis_second)} samples); "
            "profiles would not be sample-aligned"
        )
    return axis_first, first, second


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
    """Everything measured from one frame's stripes, kept per stripe.

    ``first``/``second`` are the two stripes in gap order: top then bottom for
    horizontal stripes, left then right for vertical ones. ``axis`` is the
    coordinate the profiles run along (x for horizontal stripes, y for vertical).

    Each stripe is fitted independently. ``reference`` is the single normalisation
    value shared by both -- the larger of the two fitted maxima -- so the brighter
    stripe peaks at 1.0 and the relative offset between the stripes is preserved.
    """

    detection: Detection
    orientation: str
    position: str
    reference_ring: str
    clearances: Dict[str, float]
    first_roi: Tuple[int, int, int, int]
    second_roi: Tuple[int, int, int, int]
    axis: np.ndarray
    first_profile: np.ndarray
    second_profile: np.ndarray
    first_fit: Dict[str, object]
    second_fit: Dict[str, object]
    reference: float
    first_norm: np.ndarray
    second_norm: np.ndarray
    first_fitted_norm: np.ndarray
    second_fitted_norm: np.ndarray
    axis_angle_deg: np.ndarray
    axis_mm: np.ndarray
    axis_from_center: np.ndarray
    # Stripe direction. angle 0/90 with stripes=None means the axis-aligned path.
    angle: float = 0.0
    angle_source: str = "horizontal"
    gradient: Optional[Dict[str, float]] = None
    stripes: Optional[Tuple["Stripe", "Stripe"]] = None

    @property
    def rotated(self) -> bool:
        return self.stripes is not None

    @property
    def labels(self) -> Tuple[str, str]:
        if self.rotated:
            return ("stripeA", "stripeB")
        return STRIPE_LABELS[self.orientation]

    @property
    def axis_name(self) -> str:
        if self.rotated:
            return "s"  # signed distance along the stripe from its centre
        return PROFILE_AXIS[self.orientation]

    @property
    def rois(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        return self.first_roi, self.second_roi

    @property
    def profiles(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.first_profile, self.second_profile

    @property
    def fits(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        return self.first_fit, self.second_fit

    @property
    def norms(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.first_norm, self.second_norm

    @property
    def fitted_norms(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.first_fitted_norm, self.second_fitted_norm

    @property
    def reference_label(self) -> str:
        """Which stripe's fitted maximum set the normalisation reference."""
        first_peak = float(self.first_fit["peak_value"])
        return self.labels[0] if first_peak >= float(self.second_fit["peak_value"]) else self.labels[1]


def analyse(
    detection: Detection,
    orientation: str = DEFAULT_ORIENTATION,
    length_factor: float = DEFAULT_STRIPE_LENGTH_FACTOR,
    thickness: int = DEFAULT_STRIPE_THICKNESS,
    position: str = DEFAULT_STRIPE_POSITION,
    mark_clearance: float = DEFAULT_MARK_CLEARANCE,
    offset: float = DEFAULT_STRIPE_OFFSET,
    reference_ring: str = DEFAULT_STRIPE_REFERENCE,
    object_distance_mm: float = DEFAULT_OBJECT_DISTANCE_MM,
    stripe_angle: Optional[float] = None,
    fallback_angle: float = 0.0,
    min_gradient: float = DEFAULT_MIN_GRADIENT,
    gradient_sigma: float = DEFAULT_GRADIENT_SIGMA,
) -> ProfileResult:
    """Place the stripes, then fit and normalise each stripe's profile separately.

    The two stripes are never summed. Each gets its own second-order fit, and both
    are normalised by the same reference -- the larger of the two fitted maxima --
    so the curves stay directly comparable to one another.

    ``orientation="auto"`` or an explicit ``stripe_angle`` runs stripes at that angle
    via rotated sampling; "horizontal"/"vertical" keep the axis-aligned integer-box
    path so previously published numbers stay reproducible.
    """
    chosen = resolve_stripe_angle(
        detection,
        orientation,
        stripe_angle=stripe_angle,
        fallback_angle=fallback_angle,
        min_gradient=min_gradient,
        gradient_sigma=gradient_sigma,
    )
    angle, angle_source, gradient = chosen["angle"], chosen["source"], chosen["gradient"]
    rotated = angle_source in ("fixed", "gradient", "fallback")

    if rotated:
        stripes = rotated_stripes(
            detection,
            angle,
            length_factor=length_factor,
            thickness=thickness,
            position=position,
            mark_clearance=mark_clearance,
            offset=offset,
            reference=reference_ring,
        )
        axis, first = sample_stripe(detection.image, stripes[0])
        axis_b, second = sample_stripe(detection.image, stripes[1])
        if len(axis_b) != len(axis):
            raise ValueError("rotated stripes produced different sample counts")
        labels = ("stripeA", "stripeB")
        clearances = rotated_clearances(detection, stripes, labels)
        first_roi = stripes[0].bounding_box(detection.image.shape)
        second_roi = stripes[1].bounding_box(detection.image.shape)
        # The profile axis is distance along the stripe, already centred on the
        # pattern, so angle/mm use the sensor extent along the dominant component.
        size, fov = axis_geometry(detection.image.shape, "horizontal" if abs(np.cos(np.radians(angle))) >= 0.5 else "vertical")
        axis_from_center = axis.copy()
        axis_angle = np.degrees(np.arctan(axis / _focal_px(size, fov)))
        axis_mm_values = object_distance_mm * axis / _focal_px(size, fov)
    else:
        stripes = None
        first_roi, second_roi = stripe_rois(
            detection,
            orientation=orientation,
            length_factor=length_factor,
            thickness=thickness,
            position=position,
            mark_clearance=mark_clearance,
            offset=offset,
            reference=reference_ring,
        )
        axis, first, second = stripe_profiles(detection.image, first_roi, second_roi, orientation)
        clearances = stripe_clearances(detection, (first_roi, second_roi), orientation)
        size, fov = axis_geometry(detection.image.shape, orientation)
        centre_along = detection.center[0] if orientation == "horizontal" else detection.center[1]
        axis_from_center = axis - centre_along
        axis_angle = pixel_to_angle(axis, size, fov)
        axis_mm_values = pixel_to_mm(axis, object_distance_mm, size, fov)

    first_fit = fit_second_order(axis, first)
    second_fit = fit_second_order(axis, second)
    reference = max(float(first_fit["peak_value"]), float(second_fit["peak_value"]))

    return ProfileResult(
        detection=detection,
        orientation=orientation,
        position=position,
        reference_ring=reference_ring,
        angle=angle,
        angle_source=angle_source,
        gradient=gradient,
        stripes=stripes,
        clearances=clearances,
        first_roi=first_roi,
        second_roi=second_roi,
        axis=axis,
        first_profile=first,
        second_profile=second,
        first_fit=first_fit,
        second_fit=second_fit,
        reference=reference,
        first_norm=normalize_by(first, reference),
        second_norm=normalize_by(second, reference),
        first_fitted_norm=normalize_by(first_fit["fitted"], reference),
        second_fitted_norm=normalize_by(second_fit["fitted"], reference),
        axis_angle_deg=axis_angle,
        axis_mm=axis_mm_values,
        axis_from_center=axis_from_center,
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
    """Draw the two measurement stripes, rotated when the angle is not axis-aligned."""
    if result.rotated:
        for stripe, color, label in zip(result.stripes, STRIPE_COLORS, result.labels):
            axis.add_patch(
                Polygon(stripe.corners(), closed=True, fill=False, edgecolor=color,
                        linewidth=linewidth, label=label)
            )
        return
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
    if result.rotated:
        grad = result.gradient
        extra = f", |grad| {grad['magnitude']:.2f}/px" if grad else ""
        heading = (
            f"Stripes at {result.angle:.1f} deg ({result.angle_source}{extra}), "
            f"position '{result.position}'"
        )
    else:
        heading = f"{result.orientation.title()} stripes, position '{result.position}'"
    overview.set_title(f"{heading} - {image_name}  (mark SNR {detection.inner_snr:.1f})")
    overview.legend(loc="upper right", framealpha=0.85, fontsize=9)
    overview.set_xlabel("x (px)")
    overview.set_ylabel("y (px)")

    # Zoom on the pattern to confirm each stripe clears the marks. Bounds are the
    # union of both stripes and the reference ring, so this works either way round
    # and whether or not the outer ring was detected.
    ring = result.reference_ring
    ring_marks = detection.inner_marks if ring == "inner" else detection.outer_marks
    zoom = fig.add_subplot(grid[1, 0:4])
    pad = 60
    height, width = detection.image.shape
    boxes = np.array(result.rois, dtype=np.float64)
    zx0 = max(0, int(min(boxes[:, 0].min(), ring_marks[:, 0].min())) - pad)
    zx1 = min(width - 1, int(max(boxes[:, 2].max(), ring_marks[:, 0].max())) + pad)
    zy0 = max(0, int(min(boxes[:, 1].min(), ring_marks[:, 1].min())) - pad)
    zy1 = min(height - 1, int(max(boxes[:, 3].max(), ring_marks[:, 1].max())) + pad)
    zoom.imshow(display, cmap="gray", vmin=0, vmax=255)
    _draw_marks(zoom, detection)
    _draw_stripes(zoom, result, linewidth=2.2)
    zoom.set_xlim(zx0, zx1)
    zoom.set_ylim(zy1, zy0)
    zoom.set_aspect("equal")
    # Shade the span the reference mark pairs occupy, so it is obvious whether a
    # stripe edge falls inside or outside the mark. Only meaningful axis-aligned.
    gap_axis = "y" if result.orientation == "horizontal" else "x"
    if not result.rotated:
        extent = measure_mark_extent(detection, ring, gap_axis)
        for edge in detection.edge(ring, gap_axis):
            if result.orientation == "horizontal":
                zoom.axhspan(edge - extent, edge + extent, color=RING_STYLE[ring], alpha=0.18, zorder=0)
            else:
                zoom.axvspan(edge - extent, edge + extent, color=RING_STYLE[ring], alpha=0.18, zorder=0)
        shaded = f"shaded = {ring} mark span ({2 * extent:.0f} px wide); "
    else:
        shaded = ""

    gaps = ", ".join(f"{label.split()[0]} {gap:+.1f} px" for label, gap in result.clearances.items())
    zoom.set_title(f"Stripe placement - {shaded}clearance {gaps}", fontsize=10)
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

    # Each stripe fitted on its own, then both normalised by the shared reference.
    fit_axis = fig.add_subplot(grid[3, 0:3])
    for values, fit, color, label in zip(result.profiles, result.fits, STRIPE_COLORS, result.labels):
        fit_axis.plot(result.axis, values, color=color, linewidth=1.0, alpha=0.55)
        fit_axis.plot(
            result.axis,
            fit["fitted"],
            color=color,
            linewidth=2.2,
            label=f"{label} fit  max {fit['peak_value']:.0f}  RMSE {fit['rmse']:.0f}",
        )
        fit_axis.plot(fit["peak_x"], fit["peak_value"], "o", color=color, markersize=7, markeredgecolor="#333333")
    fit_axis.set_xlabel(f"{axis_name} (px)")
    fit_axis.set_ylabel("mean intensity")
    fit_axis.set_title("Each stripe fitted separately (2nd order)", fontsize=10)
    fit_axis.legend(fontsize=8)
    fit_axis.grid(alpha=0.25)

    norm_axis = fig.add_subplot(grid[3, 3:6])
    for values, fitted, color, label in zip(result.norms, result.fitted_norms, STRIPE_COLORS, result.labels):
        norm_axis.plot(result.axis, values, color=color, linewidth=1.0, alpha=0.55)
        norm_axis.plot(result.axis, fitted, color=color, linewidth=2.2, label=f"{label} fit")
    norm_axis.axhline(1.0, color="#3ba7e0", linewidth=1.0, linestyle="--", alpha=0.8)
    norm_axis.set_xlabel(f"{axis_name} (px)")
    norm_axis.set_ylabel("normalised intensity")
    norm_axis.set_title(
        f"Normalised by max of the two fits = {result.reference:.0f} ({result.reference_label})",
        fontsize=10,
    )
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
    first, second = (label.split()[0] for label in result.labels)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                f"{axis_name}_pixels",
                f"{axis_name}_from_center_px",
                f"{axis_name}_angle_deg",
                f"{axis_name}_mm",
                first,
                f"{first}_fitted",
                f"{first}_norm",
                f"{first}_fitted_norm",
                second,
                f"{second}_fitted",
                f"{second}_norm",
                f"{second}_fitted_norm",
            ]
        )
        writer.writerows(
            zip(
                result.axis,
                result.axis_from_center,
                result.axis_angle_deg,
                result.axis_mm,
                result.first_profile,
                result.first_fit["fitted"],
                result.first_norm,
                result.first_fitted_norm,
                result.second_profile,
                result.second_fit["fitted"],
                result.second_norm,
                result.second_fitted_norm,
            )
        )


def result_to_dict(result: ProfileResult, detection_summary: Dict[str, object]) -> Dict[str, object]:
    """Merge the detection summary with the stripe ROIs and per-stripe fit parameters."""
    axis_name = result.axis_name

    def fit_dict(fit: Dict[str, object]) -> Dict[str, object]:
        return {
            "model": f"intensity = c0 + c1*{axis_name} + c2*{axis_name}^2",
            "coefficients": fit["coefficients"],
            "rmse": fit["rmse"],
            "peak_x": fit["peak_x"],
            "peak_value": fit["peak_value"],
            "vertex_x": fit["vertex_x"],
            "concave": fit["concave"],
            "vertex_inside_stripe": fit["vertex_inside"],
            "peak_is_interior": fit["peak_is_interior"],
        }

    keys = [label.replace(" ", "_") for label in result.labels]
    return {
        **detection_summary,
        "stripe_orientation": result.orientation,
        "stripe_angle_deg": result.angle,
        "stripe_angle_source": result.angle_source,
        "gradient": result.gradient,
        "stripe_position": result.position,
        "stripe_reference_ring": result.reference_ring,
        "stripe_mark_clearance_px": result.clearances,
        "profile_axis": axis_name,
        f"{keys[0]}_roi": list(result.first_roi),
        f"{keys[1]}_roi": list(result.second_roi),
        "roi_format": "x0, y0, x1, y1 (inclusive pixel bounds)",
        "profiles_summed": False,
        f"{keys[0]}_fit": fit_dict(result.first_fit),
        f"{keys[1]}_fit": fit_dict(result.second_fit),
        "normalization": {
            "reference": result.reference,
            "rule": "max of the two fitted maxima, shared by both stripes",
            "reference_from": result.reference_label,
        },
    }


def describe(result: ProfileResult) -> str:
    """Human-readable measurement summary."""
    axis_name = result.axis_name
    width = max(len(label) for label in result.labels)
    window = f"{result.axis.min():.0f}-{result.axis.max():.0f}"
    lines = [f"  stripe position: {result.position} (referenced to the {result.reference_ring} ring)"]
    if result.rotated:
        grad = result.gradient
        detail = f" from gradient |g|={grad['magnitude']:.2f}/px" if grad and grad.get("determined") else ""
        if grad is not None and not grad.get("determined"):
            detail = f" (gradient only {grad['magnitude']:.2f}/px - too weak, used fallback)"
        lines.append(f"  stripe angle: {result.angle:.1f} deg [{result.angle_source}]{detail}")

    for roi, label in zip(result.rois, result.labels):
        gap = result.clearances[label]
        state = f"{gap:.1f} px clear of nearest mark" if gap >= 0 else f"OVERLAPS a mark by {-gap:.1f} px"
        lines.append(
            f"  {label:<{width}} (x0,y0,x1,y1) = {roi}   "
            f"{roi[2] - roi[0] + 1} x {roi[3] - roi[1] + 1} px   {state}"
        )

    lines.append("  profiles fitted separately (not summed):")
    for fit, label in zip(result.fits, result.labels):
        c0, c1, c2 = fit["coefficients"]
        lines.append(
            f"    {label:<{width}} {c0:.5g} + {c1:.5g}*{axis_name} + {c2:.5g}*{axis_name}^2   "
            f"RMSE {fit['rmse']:.0f}   max {fit['peak_value']:.1f} at {axis_name}={fit['peak_x']:.0f}"
        )
        if not fit["peak_is_interior"]:
            reason = (
                f"convex (c2 = {c2:+.4g}), no maximum; profile monotonic"
                if not fit["concave"]
                else f"vertex at {axis_name} = {fit['vertex_x']:.0f} outside the stripe ({window})"
            )
            lines.append(f"      NOTE: {reason}; its fitted max is a window edge")

    lines.append(
        f"  normalised by max of the two fits = {result.reference:.1f}  (from the {result.reference_label})"
    )
    return "\n".join(lines)


# ------------------------------------------------------------------------- driver


def process_image(
    image_path: Path,
    output_dir: Path,
    detection_kwargs: Optional[Dict[str, object]] = None,
    orientation: str = DEFAULT_ORIENTATION,
    stripe_length_factor: float = DEFAULT_STRIPE_LENGTH_FACTOR,
    stripe_thickness: int = DEFAULT_STRIPE_THICKNESS,
    stripe_position: str = DEFAULT_STRIPE_POSITION,
    mark_clearance: float = DEFAULT_MARK_CLEARANCE,
    stripe_offset: float = DEFAULT_STRIPE_OFFSET,
    stripe_reference: str = DEFAULT_STRIPE_REFERENCE,
    object_distance_mm: float = DEFAULT_OBJECT_DISTANCE_MM,
    stripe_angle: Optional[float] = None,
    fallback_angle: float = 0.0,
    min_gradient: float = DEFAULT_MIN_GRADIENT,
    channel: str = DEFAULT_CHANNEL,
    show: bool = False,
    quiet: bool = False,
) -> Optional[Path]:
    """Detect, measure, and plot one frame. Returns the plot path."""
    image = load_image(image_path, channel=channel)
    detection = detect(image, **(detection_kwargs or {}))
    detection.channel = channel
    if not quiet:
        print(describe_detection(detection, image_path.name))

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{image_path.stem}_profile.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    json_path = output_dir / f"{image_path.stem}_profile.json"
    csv_path = output_dir / f"{image_path.stem}_profile.csv"

    summary = detection_to_dict(detection, image_path)

    # Outer-referenced stripes need both rings; inner-referenced ones need only the
    # inner ring, so a frame with a missing outer ring can still be measured.
    have_rings = len(detection.inner_marks) == 4 and (
        stripe_reference == "inner" or len(detection.outer_marks) == 4
    )
    if not have_rings:
        fig = plot_no_pattern(detection, image_path.name, plot_path)
        save_overlay(None, detection, overlay_path)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        if show:
            plt.show()
        plt.close(fig)
        reason = (
            "inner ring found but no outer ring; retry with --stripe-reference inner"
            if len(detection.inner_marks) == 4
            else "no fiducial pattern"
        )
        if not quiet:
            print(f"  saved {plot_path} (no stripes placed: {reason})")
        return plot_path

    result = analyse(
        detection,
        orientation=orientation,
        length_factor=stripe_length_factor,
        thickness=stripe_thickness,
        position=stripe_position,
        mark_clearance=mark_clearance,
        offset=stripe_offset,
        reference_ring=stripe_reference,
        object_distance_mm=object_distance_mm,
        stripe_angle=stripe_angle,
        fallback_angle=fallback_angle,
        min_gradient=min_gradient,
    )
    if not quiet:
        print(describe(result))

    fig = plot_results(result, image_path.name, plot_path)
    save_overlay(result, detection, overlay_path)
    save_profile_csv(csv_path, result)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result_to_dict(result, summary), handle, indent=2)

    if show:
        plt.show()
    plt.close(fig)

    if not quiet:
        print(f"  saved {plot_path}")
    return plot_path


def _worker(payload):
    """Module-level entry point for the process pool."""
    path, kwargs = payload
    try:
        process_image(Path(path), quiet=True, **kwargs)
        return (path, None)
    except Exception as exc:  # keep a long batch alive
        return (path, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, normalise, and plot fiducial-referenced intensity profiles from DNG frames"
    )
    parser.add_argument("input", type=Path, help="Path to a DNG file or a directory of DNG files")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for outputs")
    parser.add_argument(
        "--orientation",
        choices=ORIENTATION_CHOICES,
        default=DEFAULT_ORIENTATION,
        help=(
            "Stripe orientation: 'horizontal' profiles along x, 'vertical' along y, "
            "'auto' measures the intensity gradient across the outer circle and runs "
            "the stripes along the steepest direction"
        ),
    )
    parser.add_argument(
        "--stripe-angle",
        type=float,
        default=None,
        help="Fixed stripe long-axis angle in degrees (0 = +x). Overrides --orientation",
    )
    parser.add_argument(
        "--fallback-angle",
        type=float,
        default=0.0,
        help="Angle used when --orientation auto finds too weak a gradient to define one",
    )
    parser.add_argument(
        "--min-gradient",
        type=float,
        default=DEFAULT_MIN_GRADIENT,
        help="Gradient magnitude (counts/px) below which the auto direction is undetermined",
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
        "--stripe-position",
        choices=STRIPE_POSITIONS,
        default=DEFAULT_STRIPE_POSITION,
        help=(
            "Where each stripe sits across its ring gap: 'centred' midway between the rings, "
            "'outer' inside the gap against the outer marks, 'beyond' outside the outer ring"
        ),
    )
    parser.add_argument(
        "--mark-clearance",
        type=float,
        default=DEFAULT_MARK_CLEARANCE,
        help="Pixels to keep between a stripe edge and the nearest mark's measured arm tip",
    )
    parser.add_argument(
        "--stripe-offset",
        type=float,
        default=DEFAULT_STRIPE_OFFSET,
        help="Extra outward shift in px applied to both stripes, away from the pattern centre",
    )
    parser.add_argument(
        "--stripe-reference",
        choices=STRIPE_REFERENCES,
        default=DEFAULT_STRIPE_REFERENCE,
        help=(
            "Ring the stripes are positioned against and that sets their length. "
            "'inner' allows frames whose outer ring is missing, but those results are "
            "not comparable with outer-referenced ones"
        ),
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
    add_loading_arguments(parser)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes; >1 suppresses the per-frame detail and prints progress only",
    )
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

    shared = dict(
        output_dir=args.output_dir.resolve(),
        detection_kwargs=detection_kwargs,
        orientation=args.orientation,
        stripe_length_factor=args.stripe_length_factor,
        stripe_thickness=args.stripe_thickness,
        stripe_position=args.stripe_position,
        mark_clearance=args.mark_clearance,
        stripe_offset=args.stripe_offset,
        stripe_reference=args.stripe_reference,
        object_distance_mm=args.object_distance_mm,
        stripe_angle=args.stripe_angle,
        fallback_angle=args.fallback_angle,
        min_gradient=args.min_gradient,
        channel=args.channel,
    )

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        failures = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (path, err) in enumerate(
                pool.map(_worker, [(str(p), shared) for p in images], chunksize=1), 1
            ):
                if err:
                    failures.append((path, err))
                if i % 20 == 0 or i == len(images):
                    print(f"  {i}/{len(images)} frames", flush=True)
        for path, err in failures:
            print(f"  FAILED {Path(path).name}: {err}")
        print(f"done: {len(images) - len(failures)}/{len(images)} frames written")
        return

    for image_path in images:
        process_image(image_path, show=args.show, **shared)


if __name__ == "__main__":
    main()
