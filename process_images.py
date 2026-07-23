#!/usr/bin/env python3
"""Process DNG image data by extracting a 1D intensity profile from a user-selected ROI.

For each input image, the script can:
1. Use a bounding box to select a region of interest.
2. Compute the average or median intensity value across the y-axis for each x.
3. Smooth the profile with a low-pass filter.
4. Compute the gradient of the smoothed profile.
5. Save the profile data and a plot for each image.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rawpy

try:
    import tkinter as tk
    from PIL import Image as PILImage
    from PIL import ImageTk as PILImageTk
except Exception:  # pragma: no cover - tkinter/Pillow may be unavailable in some environments
    tk = None
    PILImage = None
    PILImageTk = None


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

    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    return gray


def parse_bbox(bbox_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if bbox_str is None:
        return None
    values = [int(v.strip()) for v in bbox_str.split(",")]
    if len(values) != 4:
        raise ValueError("Bounding box must be given as x0,x1,y0,y1")
    x0, x1, y0, y1 = values
    return (x0, x1, y0, y1)


def map_display_to_image_coords(x: int, y: int, display_width: int, display_height: int, image_width: int, image_height: int) -> Tuple[int, int]:
    """Convert coordinates from the resized display window back to the original image coordinates."""
    scale_x = image_width / display_width
    scale_y = image_height / display_height
    return int(round(x * scale_x)), int(round(y * scale_y))


def select_bbox(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Interactively let the user draw a rectangle on the image using a Tk window."""
    if tk is None or PILImage is None or PILImageTk is None:
        print("Tkinter/Pillow not available; falling back to the full image.")
        return (0, image.shape[1] - 1, 0, image.shape[0] - 1)

    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max > image_min:
        normalized = (image - image_min) / (image_max - image_min)
    else:
        normalized = np.zeros_like(image, dtype=np.float32)

    gray_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    pil_image = PILImage.fromarray(gray_uint8, mode="L")
    width, height = pil_image.size

    max_width = 1200
    max_height = 800
    scale = min(1.0, max_width / width, max_height / height)
    display_width = max(500, int(width * scale))
    display_height = max(400, int(height * scale))

    root = tk.Tk()
    root.title("Select ROI")
    root.geometry(f"{display_width}x{display_height + 50}")

    canvas = tk.Canvas(root, width=display_width, height=display_height, bg="black")
    canvas.pack()

    resized_image = pil_image.resize((display_width, display_height), PILImage.Resampling.LANCZOS)
    photo = PILImageTk.PhotoImage(resized_image)
    canvas.create_image(0, 0, anchor="nw", image=photo)

    bbox: Optional[Tuple[int, int, int, int]] = None
    start_xy: Optional[Tuple[int, int]] = None
    rect_id: Optional[int] = None

    def update_rect(x1: int, y1: int, x2: int, y2: int) -> None:
        nonlocal rect_id
        if rect_id is not None:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def on_mouse_down(event) -> None:
        nonlocal start_xy
        start_xy = (event.x, event.y)

    def on_mouse_move(event) -> None:
        nonlocal start_xy
        if start_xy is None:
            return
        x0, y0 = start_xy
        update_rect(x0, y0, event.x, event.y)

    def on_mouse_up(event) -> None:
        nonlocal bbox, start_xy
        if start_xy is None:
            return
        x0, y0 = start_xy
        x1, y1 = event.x, event.y
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0 = map_display_to_image_coords(x0, y0, display_width, display_height, width, height)
        x1, y1 = map_display_to_image_coords(x1, y1, display_width, display_height, width, height)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        if x1 - x0 > 2 and y1 - y0 > 2:
            bbox = (x0, x1, y0, y1)
        else:
            bbox = None
        start_xy = None

    def confirm_selection() -> None:
        root.destroy()

    def cancel_selection() -> None:
        nonlocal bbox
        bbox = None
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    root.bind("<Return>", lambda _event: confirm_selection())
    root.bind("<Escape>", lambda _event: cancel_selection())

    button_frame = tk.Frame(root)
    button_frame.pack(fill="x", pady=4)
    tk.Button(button_frame, text="Confirm", command=confirm_selection).pack(side="left", padx=6)
    tk.Button(button_frame, text="Cancel", command=cancel_selection).pack(side="left")

    root.mainloop()

    if bbox is None:
        print("No bounding box selected; using the full image.")
        return (0, image.shape[1] - 1, 0, image.shape[0] - 1)

    print(f"Selected bounding box: x0={bbox[0]}, x1={bbox[1]}, y0={bbox[2]}, y1={bbox[3]}")
    return bbox


def compute_profile(image: np.ndarray, bbox: Tuple[int, int, int, int], stat: str = "mean") -> np.ndarray:
    """Collapse the ROI along the y-axis to create a 1D profile across x."""
    x0, x1, y0, y1 = bbox
    roi = image[y0 : y1 + 1, x0 : x1 + 1]
    if stat == "mean":
        return np.mean(roi, axis=0)
    if stat == "median":
        return np.median(roi, axis=0)
    raise ValueError(f"Unsupported stat: {stat}")


def smooth_and_gradient(profile: np.ndarray, polyorder: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a low-order polynomial trend line and return its derivative."""
    if len(profile) < polyorder + 1:
        return profile, np.zeros_like(profile)

    x = np.arange(len(profile), dtype=np.float64)
    coeffs = np.polyfit(x, profile, polyorder)
    fitted = np.polyval(coeffs, x)
    gradient = np.gradient(fitted)
    return fitted, gradient


def pixel_to_angle(u: np.ndarray, W: int = 1920, H: int = 1440) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pixel coordinates to off-axis angles in degrees."""
    fx = (W / 2) / math.tan(math.radians(5.6 / 2))
    fy = (H / 2) / math.tan(math.radians(4.2 / 2))
    theta_x = np.degrees(np.arctan((u - W / 2) / fx))
    theta_y = np.degrees(np.arctan((u - H / 2) / fy))
    return theta_x, theta_y


def pixel_to_mm(u: np.ndarray, d: float, W: int = 1920, H: int = 1440) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pixel coordinates to object-plane coordinates in mm at distance d."""
    fx = (W / 2) / math.tan(math.radians(5.6 / 2))
    fy = (H / 2) / math.tan(math.radians(4.2 / 2))
    x_mm = d * (u - W / 2) / fx
    y_mm = d * (u - H / 2) / fy
    return x_mm, y_mm


def save_profile_csv(
    path: Path,
    x_values: np.ndarray,
    profile: np.ndarray,
    smoothed: np.ndarray,
    gradient: np.ndarray,
    x_angle: np.ndarray,
    x_mm: np.ndarray,
) -> None:
    norm_value = max(float(np.max(smoothed)), 1e-12)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_pixels", "x_angle_deg", "x_mm", "profile", "profile_norm", "smoothed", "smoothed_norm", "gradient"])
        for x, angle, mm, p, s, g in zip(x_values, x_angle, x_mm, profile, smoothed, gradient):
            writer.writerow([x, angle, mm, p, p / norm_value, s, s / norm_value, g])


def plot_profile(
    output_path: Path,
    x_values: np.ndarray,
    profile: np.ndarray,
    smoothed: np.ndarray,
    gradient: np.ndarray,
    stat: str,
    x_angle: np.ndarray,
    x_mm: np.ndarray,
) -> None:
    norm_value = max(float(np.max(smoothed)), 1e-12)
    profile_norm = profile / norm_value
    smoothed_norm = smoothed / norm_value

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    axes[0].plot(x_values, profile_norm, label=f"{stat.title()} profile (normalized)", linewidth=1.5)
    axes[0].plot(x_values, smoothed_norm, label="Low-pass filtered (normalized)", linewidth=2)
    axes[0].set_ylabel("Normalized intensity")
    axes[0].set_title("Normalized intensity profile")
    axes[0].legend(loc="best")

    axes[1].plot(x_angle, profile_norm, label=f"{stat.title()} profile (normalized)", linewidth=1.5)
    axes[1].plot(x_angle, smoothed_norm, label="Low-pass filtered (normalized)", linewidth=2)
    axes[1].set_ylabel("Normalized intensity")
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_title("Intensity vs angle")
    axes[1].legend(loc="best")

    axes[2].plot(x_mm, profile_norm, label=f"{stat.title()} profile (normalized)", linewidth=1.5)
    axes[2].plot(x_mm, smoothed_norm, label="Low-pass filtered (normalized)", linewidth=2)
    axes[2].set_ylabel("Normalized intensity")
    axes[2].set_xlabel("Position (mm)")
    axes[2].set_title("Intensity vs object-plane distance")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def process_image(
    image_path: Path,
    output_dir: Path,
    stat: str,
    bbox: Optional[Tuple[int, int, int, int]],
    show: bool,
    object_distance_mm: float,
) -> Path:
    image = load_image(image_path)

    if bbox is None:
        bbox = select_bbox(image)

    profile = compute_profile(image, bbox, stat=stat)
    smoothed, gradient = smooth_and_gradient(profile)

    x_values = np.arange(len(profile))
    image_width = image.shape[1]
    image_height = image.shape[0]
    x_angle, _ = pixel_to_angle(x_values, W=image_width, H=image_height)
    x_mm, _ = pixel_to_mm(x_values, d=object_distance_mm, W=image_width, H=image_height)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{image_path.stem}_{stat}.csv"
    plot_path = output_dir / f"{image_path.stem}_{stat}.png"

    save_profile_csv(csv_path, x_values, profile, smoothed, gradient, x_angle, x_mm)
    plot_profile(plot_path, x_values, profile, smoothed, gradient, stat, x_angle, x_mm)

    if show:
        plt.show()
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        norm_value = max(float(np.max(smoothed)), 1e-12)
        profile_norm = profile / norm_value
        smoothed_norm = smoothed / norm_value
        axes[0].plot(x_values, profile_norm, label=f"{stat.title()} profile (normalized)", linewidth=1.5)
        axes[0].plot(x_values, smoothed_norm, label="Low-pass filtered (normalized)", linewidth=2)
        axes[0].set_ylabel("Normalized intensity")
        axes[0].set_title("Normalized intensity profile")
        axes[0].legend(loc="best")
        axes[1].plot(x_angle, profile_norm, label=f"{stat.title()} profile (normalized)", linewidth=1.5)
        axes[1].plot(x_angle, smoothed_norm, label="Low-pass filtered (normalized)", linewidth=2)
        axes[1].set_ylabel("Normalized intensity")
        axes[1].set_xlabel("Angle (deg)")
        axes[1].set_title("Intensity vs angle")
        axes[1].legend(loc="best")
        axes[2].plot(x_mm, profile_norm, label=f"{stat.title()} profile (normalized)", linewidth=1.5)
        axes[2].plot(x_mm, smoothed_norm, label="Low-pass filtered (normalized)", linewidth=2)
        axes[2].set_ylabel("Normalized intensity")
        axes[2].set_xlabel("Position (mm)")
        axes[2].set_title("Intensity vs object-plane distance")
        axes[2].legend(loc="best")
        fig.tight_layout()
        plt.show()

    return csv_path


def discover_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.dng")) + sorted(input_path.glob("*.DNG"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and smooth intensity profiles from DNG images")
    parser.add_argument("input", type=Path, help="Path to a DNG file or a directory of DNG files")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory to save plots and CSV files")
    parser.add_argument("--stat", choices=["mean", "median"], default="mean", help="Aggregation method across the y-axis")
    parser.add_argument("--bbox", type=str, default=None, help="Optional rectangle as x0,x1,y0,y1 to skip interactive selection")
    parser.add_argument("--show", action="store_true", help="Display the plot after processing")
    parser.add_argument("--object-distance-mm", type=float, default=450.0, help="Object distance in mm used for the mm-axis conversion")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    bbox = parse_bbox(args.bbox)

    images = discover_images(input_path)
    if not images:
        raise FileNotFoundError(f"No DNG files found in {input_path}")

    for image_path in images:
        csv_path = process_image(image_path, output_dir, args.stat, bbox, args.show, args.object_distance_mm)
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
