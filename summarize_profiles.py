#!/usr/bin/env python3
"""Summarise stripe profiles across a batch of DNG frames.

Runs detect_fiducials + process_images over many frames and collects one row per
file, without writing the per-frame figures -- for a few hundred frames those cost
minutes and gigabytes, and the point here is the table.

Reported per stripe are the max and min of its *normalised* profile, where both
stripes share one reference (the larger of the two fitted maxima), so the numbers
are comparable between stripes within a frame.

    python summarize_profiles.py <dir> --glob '*cam_2*' --orientation vertical \
        --output output/summary_x_axis.csv
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import process_images as P
from detect_fiducials import add_loading_arguments, detect_file, discover_images


@dataclass
class Row:
    """One frame's summary."""

    name: str
    ok: bool
    note: str = ""
    outer_source: str = ""
    outer_measured: int = 0
    snr: float = float("nan")
    r_inner: float = float("nan")
    r_outer: float = float("nan")
    ring_ratio: float = float("nan")
    mm_per_px: float = float("nan")
    centre_x: float = float("nan")
    centre_y: float = float("nan")
    labels: Optional[List[str]] = None
    norm_max: Optional[List[float]] = None
    norm_min: Optional[List[float]] = None
    fit_norm_max: Optional[List[float]] = None
    fit_norm_min: Optional[List[float]] = None
    reference: float = float("nan")
    reference_from: str = ""
    angle: float = float("nan")
    angle_source: str = ""
    gradient_mag: float = float("nan")


def summarise_one(args) -> Row:
    """Detect, measure, and reduce one frame to a summary row."""
    path, orientation, options = args
    path = Path(path)
    try:
        detection = detect_file(path, **options.get("detect", {}))
    except Exception as exc:  # pragma: no cover - defensive, keeps a batch running
        return Row(name=path.name, ok=False, note=f"detect failed: {exc}")

    if not detection.found:
        note = "inner ring only" if len(detection.inner_marks) == 4 else "no pattern"
        return Row(name=path.name, ok=False, note=note, snr=detection.inner_snr)

    try:
        result = P.analyse(detection, orientation=orientation, **options.get("analyse", {}))
    except Exception as exc:
        return Row(name=path.name, ok=False, note=f"analyse failed: {exc}", snr=detection.inner_snr)

    return Row(
        name=path.name,
        ok=True,
        outer_source=detection.outer_source,
        outer_measured=detection.outer_detected,
        snr=detection.inner_snr,
        r_inner=detection.inner_radius,
        r_outer=detection.outer_radius,
        ring_ratio=detection.ring_ratio_measured,
        mm_per_px=detection.mm_per_px,
        centre_x=detection.center[0],
        centre_y=detection.center[1],
        labels=[label.split()[0] for label in result.labels],
        norm_max=[float(v.max()) for v in result.norms],
        norm_min=[float(v.min()) for v in result.norms],
        fit_norm_max=[float(v.max()) for v in result.fitted_norms],
        fit_norm_min=[float(v.min()) for v in result.fitted_norms],
        reference=result.reference,
        reference_from=result.reference_label,
        angle=result.angle,
        angle_source=result.angle_source,
        gradient_mag=(result.gradient or {}).get("magnitude", float("nan")),
    )


def write_csv(path: Path, rows: List[Row], labels: List[str]) -> None:
    header = ["file", "status", "note", "outer_ring_source", "outer_marks_measured", "mark_snr"]
    header += ["r_inner_px", "r_outer_px", "ring_ratio_measured", "mm_per_px", "centre_x", "centre_y"]
    for label in labels:
        header += [f"{label}_norm_max", f"{label}_norm_min", f"{label}_fit_norm_max", f"{label}_fit_norm_min"]
    header += ["normalisation_reference", "reference_from", "stripe_angle_deg", "angle_source", "gradient_magnitude"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for r in rows:
            row = [r.name, "ok" if r.ok else "skipped", r.note, r.outer_source, r.outer_measured, r.snr]
            row += [r.r_inner, r.r_outer, r.ring_ratio, r.mm_per_px, r.centre_x, r.centre_y]
            for i in range(len(labels)):
                if r.ok and r.norm_max is not None:
                    row += [r.norm_max[i], r.norm_min[i], r.fit_norm_max[i], r.fit_norm_min[i]]
                else:
                    row += ["", "", "", ""]
            row += [r.reference, r.reference_from, r.angle, r.angle_source, r.gradient_mag]
            writer.writerow(row)


def print_table(rows: List[Row], labels: List[str], limit: Optional[int] = None) -> None:
    """Per-file table of normalised max/min for each stripe."""
    head = f"{'file':<44} {'src':<9}"
    for label in labels:
        head += f" {label + '_max':>10} {label + '_min':>10}"
    print(head)
    print("-" * len(head))
    shown = rows if limit is None else rows[:limit]
    for r in shown:
        if not r.ok:
            print(f"{r.name:<44} {'SKIP':<9}  {r.note}")
            continue
        line = f"{r.name:<44} {r.outer_source:<9}"
        for i in range(len(labels)):
            line += f" {r.norm_max[i]:>10.4f} {r.norm_min[i]:>10.4f}"
        print(line)
    if limit is not None and len(rows) > limit:
        print(f"... {len(rows) - limit} more rows (see the CSV)")


def print_stats(rows: List[Row], labels: List[str]) -> None:
    """Aggregate statistics over the frames that produced a measurement."""
    good = [r for r in rows if r.ok]
    print(f"\nframes: {len(rows)}   measured: {len(good)}   skipped: {len(rows) - len(good)}")
    if not good:
        return

    src: Dict[str, int] = {}
    for r in good:
        src[r.outer_source] = src.get(r.outer_source, 0) + 1
    print(f"outer ring source: {src}")

    def stat(values, fmt="%.4f"):
        v = np.array([x for x in values if np.isfinite(x)], dtype=np.float64)
        if not len(v):
            return "n/a"
        return f"{fmt % v.mean()} +- {fmt % v.std()}  [{fmt % v.min()} .. {fmt % v.max()}]"

    print(f"mark SNR          : {stat([r.snr for r in good], '%.1f')}")
    print(f"r_inner (px)      : {stat([r.r_inner for r in good], '%.2f')}")
    print(f"r_outer (px)      : {stat([r.r_outer for r in good], '%.2f')}")
    print(f"ring ratio        : {stat([r.ring_ratio for r in good])}")
    print(f"mm per px         : {stat([r.mm_per_px for r in good], '%.5f')}")
    for i, label in enumerate(labels):
        print(f"{label + ' norm max':<18}: {stat([r.norm_max[i] for r in good])}")
        print(f"{label + ' norm min':<18}: {stat([r.norm_min[i] for r in good])}")
    ref_from: Dict[str, int] = {}
    for r in good:
        ref_from[r.reference_from] = ref_from.get(r.reference_from, 0) + 1
    print(f"normalisation reference taken from: {ref_from}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise stripe profiles across a batch of DNG frames")
    parser.add_argument("input", type=Path, help="Directory (or file) of DNG frames")
    parser.add_argument("--glob", type=str, default=None, help="Filename glob filter, e.g. '*cam_2*'")
    add_loading_arguments(parser)
    parser.add_argument("--orientation", choices=P.ORIENTATION_CHOICES, default=P.DEFAULT_ORIENTATION)
    parser.add_argument("--stripe-angle", type=float, default=None, help="Fixed stripe angle in deg (0 = +x)")
    parser.add_argument("--fallback-angle", type=float, default=0.0, help="Angle used when auto finds no gradient")
    parser.add_argument("--min-gradient", type=float, default=P.DEFAULT_MIN_GRADIENT)
    parser.add_argument("--stripe-position", choices=P.STRIPE_POSITIONS, default=P.DEFAULT_STRIPE_POSITION)
    parser.add_argument("--stripe-length-factor", type=float, default=P.DEFAULT_STRIPE_LENGTH_FACTOR)
    parser.add_argument("--stripe-thickness", type=int, default=P.DEFAULT_STRIPE_THICKNESS)
    parser.add_argument("--mark-clearance", type=float, default=P.DEFAULT_MARK_CLEARANCE)
    parser.add_argument("--output", type=Path, required=True, help="CSV path for the per-file summary")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N frames")
    parser.add_argument("--print-rows", type=int, default=None, help="Print at most N table rows")
    args = parser.parse_args()

    files = discover_images(args.input.resolve(), args.glob)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No DNG files matched in {args.input}")

    options = {
        "detect": {"channel": args.channel},
        "analyse": {
            "position": args.stripe_position,
            "length_factor": args.stripe_length_factor,
            "thickness": args.stripe_thickness,
            "mark_clearance": args.mark_clearance,
            "stripe_angle": args.stripe_angle,
            "fallback_angle": args.fallback_angle,
            "min_gradient": args.min_gradient,
        },
    }
    if args.stripe_angle is not None or args.orientation == "auto":
        labels = ["stripeA", "stripeB"]
    else:
        labels = [label.split()[0] for label in P.STRIPE_LABELS[args.orientation]]

    print(
        f"{len(files)} frames from {args.input}  |  {args.orientation} stripes, "
        f"position '{args.stripe_position}'  |  {args.workers} workers"
    )
    payload = [(str(f), args.orientation, options) for f in files]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(summarise_one, payload, chunksize=1))

    print_table(rows, labels, limit=args.print_rows)
    print_stats(rows, labels)
    write_csv(args.output, rows, labels)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
