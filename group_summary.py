#!/usr/bin/env python3
"""Group a per-file profile summary into stage positions and average each group.

Frames are captured in bursts, one burst per stage position, so a gap in capture
time separates positions. This reads the ungrouped CSV written by
summarize_profiles.py and prints a per-position view of it; the CSV itself is left
alone.

    python group_summary.py output/summary_45cm_x_axis_horizontal.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

# Stripe label pairs, keyed by the columns present in the CSV. "stripeA/B" is what
# the rotated (auto-gradient) path emits, since neither axis name applies.
LABEL_PAIRS = (("top", "bottom"), ("left", "right"), ("stripeA", "stripeB"))


def capture_seconds(filename: str) -> float:
    """Seconds-of-day from a ``YYYYmmdd_HHMMSS_mmm_...`` filename."""
    stamp = filename[9:22]
    return int(stamp[0:2]) * 3600 + int(stamp[2:4]) * 60 + int(stamp[4:6]) + int(stamp[7:10]) / 1000.0


def detect_labels(fieldnames: List[str]) -> tuple:
    for pair in LABEL_PAIRS:
        if f"{pair[0]}_norm_max" in fieldnames:
            return pair
    raise ValueError(f"no known stripe columns in {fieldnames}")


def group_by_gap(rows: List[dict], gap: float) -> List[List[dict]]:
    groups = [[rows[0]]]
    for previous, row in zip(rows, rows[1:]):
        if row["_t"] - previous["_t"] > gap:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a per-position view of a profile summary CSV")
    parser.add_argument("csv_files", type=Path, nargs="+", help="Summary CSVs from summarize_profiles.py")
    parser.add_argument("--gap", type=float, default=8.0, help="Capture-time gap in s that starts a new position")
    parser.add_argument("--out-text", type=Path, default=None, help="Also write the tables to this text file")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Also write one combined grouped CSV (a row per position, tagged with its set)",
    )
    args = parser.parse_args()

    lines: List[str] = []
    grouped_rows: List[dict] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    for path in args.csv_files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            labels = detect_labels(reader.fieldnames or [])
        for row in rows:
            row["_t"] = capture_seconds(row["file"])
        rows.sort(key=lambda r: r["_t"])
        groups = group_by_gap(rows, args.gap)

        # Report the max and min of each stripe's *fitted* normalised curve: the fit
        # is what the measurement is, and it is not dominated by per-pixel noise the
        # way the raw extremes are. Each position is a burst, so also show how the
        # value varies across it.
        metrics = [f"{label}_fit_norm_{kind}" for label in labels for kind in ("max", "min")]

        emit(f"=== {path.name} : {len(rows)} files, {len(groups)} position groups ===")
        emit(f"{'':<24}" + "".join(f"{m.replace('_fit_norm', ''):^26}" for m in metrics))
        header = f"{'pos':<4} {'n':<4} {'time span':<15}" + "".join(
            f"{'avg':>8}{'min':>9}{'max':>9}" for _ in metrics
        )
        emit(header)
        emit("-" * len(header))

        for index, group in enumerate(groups, 1):
            usable = [r for r in group if r["status"] == "ok"]
            stamp = f"{group[0]['file'][9:15]}-{group[-1]['file'][9:15]}"
            if not usable:
                emit(f"{index:<4} {len(group):<4} {stamp:<15}  all {len(group)} frames skipped (no pattern)")
                continue

            record = {
                "set": path.stem,
                "position": index,
                "n_frames": len(group),
                "n_used": len(usable),
                "time_start": group[0]["file"][9:22],
                "time_end": group[-1]["file"][9:22],
            }
            cells = []
            for metric in metrics:
                values = np.array([float(r[metric]) for r in usable], dtype=np.float64)
                cells += [values.mean(), values.min(), values.max()]
                record[f"{metric}_avg"] = round(float(values.mean()), 6)
                record[f"{metric}_min"] = round(float(values.min()), 6)
                record[f"{metric}_max"] = round(float(values.max()), 6)

            note = "" if len(usable) == len(group) else f"  ({len(group) - len(usable)} skipped)"
            emit(
                f"{index:<4} {len(usable):<4} {stamp:<15}"
                + "".join(f"{v:>8.4f}{cells[i * 3 + 1]:>9.4f}{cells[i * 3 + 2]:>9.4f}"
                          for i, v in enumerate(cells[::3]))
                + note
            )
            grouped_rows.append(record)

        emit()

    if args.out_text:
        args.out_text.parent.mkdir(parents=True, exist_ok=True)
        args.out_text.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out_text}")

    if args.out_csv and grouped_rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(grouped_rows[0].keys())
        for row in grouped_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(grouped_rows)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
