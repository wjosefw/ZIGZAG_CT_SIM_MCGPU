#!/usr/bin/env python3
"""
Build listmode data from MC-GPU simulation results.

For each detector pixel in each projection and sweep, outputs one row:
    source_x, source_y, source_z, det_x, det_y, det_z, pixel_value

Source positions and directions are parsed directly from MC-GPU output headers
(Option A), so they match exactly what the simulation used.

Detector pixel world positions are computed from:
    det_center = source + direction * SDD
    pixel_world = det_center + local_x * X_det + local_z * Z_det

Where X_det and Z_det are the detector plane axes derived from the beam
direction and the rotation axis.

Output is saved as a float32 numpy binary file (.npy).
"""

import numpy as np
import glob
import re
import os
import argparse


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_header(filepath):
    """Parse an MC-GPU output header file for per-projection geometry."""
    with open(filepath, "r") as f:
        text = f.read()

    m = re.search(
        r"projection\s+(\d+)\s+of\s+(\d+).*?angle\s*=\s*([\d.]+)", text
    )
    proj_num = int(m.group(1))
    total_proj = int(m.group(2))
    angle_deg = float(m.group(3))

    m = re.search(
        r"Focal spot position = \(([-\d.]+),([-\d.]+),([-\d.]+)\).*?"
        r"direction = \(([-\d.]+),([-\d.]+),([-\d.]+)\)",
        text,
    )
    source_pos = np.array([float(m.group(i)) for i in range(1, 4)])
    direction = np.array([float(m.group(i)) for i in range(4, 7)])

    m = re.search(r"Number of pixels in X and Z:\s+(\d+)\s+(\d+)", text)
    nx, nz = int(m.group(1)), int(m.group(2))

    return {
        "proj_num": proj_num,
        "total_proj": total_proj,
        "angle_deg": angle_deg,
        "source_pos": source_pos,
        "direction": direction,
        "nx": nx,
        "nz": nz,
    }


def parse_in_file(filepath):
    """Parse an MC-GPU .in file for detector geometry parameters."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    def first_token(line):
        return line.split("#")[0].strip().split()

    # Line 41: IMAGE SIZE (width, height) Dx Dz [cm]
    tokens = first_token(lines[40])
    width_x, height_z = float(tokens[0]), float(tokens[1])

    # Line 42: SOURCE-TO-DETECTOR DISTANCE
    tokens = first_token(lines[41])
    sdd = float(tokens[0])

    # Line 43: IMAGE OFFSET
    tokens = first_token(lines[42])
    offset_x, offset_z = float(tokens[0]), float(tokens[1])

    # Line 59: AXIS OF ROTATION
    tokens = first_token(lines[58])
    rot_axis = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])])
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    return {
        "sdd": sdd,
        "width_x": width_x,
        "height_z": height_z,
        "offset_x": offset_x,
        "offset_z": offset_z,
        "rot_axis": rot_axis,
    }


# ---------------------------------------------------------------------------
# Detector geometry
# ---------------------------------------------------------------------------

def compute_detector_pixels(source_pos, direction, sdd, width_x, height_z,
                            nx, nz, offset_x, offset_z, rot_axis):
    """
    Compute world coordinates of every detector pixel center.

    Returns array of shape (nz, nx, 3) matching the MC-GPU raw file layout
    (Z-slow, X-fast).
    """
    det_center = source_pos + direction * sdd

    # Detector plane axes
    # MC-GPU azimuthal direction is cross(direction, rot_axis) — opposite sign
    # to cross(rot_axis, direction).
    x_det = np.cross(direction, rot_axis)
    x_det /= np.linalg.norm(x_det)
    z_det = rot_axis.copy()

    # Local pixel coordinates (center of each pixel)
    pix_size_x = width_x / nx
    pix_size_z = height_z / nz

    local_x = (np.arange(nx) + 0.5) * pix_size_x - 0.5 * width_x + offset_x
    local_z = (np.arange(nz) + 0.5) * pix_size_z - 0.5 * height_z + offset_z

    # Broadcast to (nz, nx, 3)
    # ii shape (nz, nx), jj shape (nz, nx)
    ii, jj = np.meshgrid(local_x, local_z)

    positions = (
        det_center[np.newaxis, np.newaxis, :]
        + ii[:, :, np.newaxis] * x_det[np.newaxis, np.newaxis, :]
        + jj[:, :, np.newaxis] * z_det[np.newaxis, np.newaxis, :]
    )
    return positions  # (nz, nx, 3)


# ---------------------------------------------------------------------------
# Raw file reading
# ---------------------------------------------------------------------------

def read_raw_projection(filepath, nx, nz, signal_channel="total"):
    """
    Read one image channel from an MC-GPU .raw file.

    Each .raw file contains two images back-to-back:
      - channel 0: total signal (scatter + primaries)
      - channel 1: primaries only

    Returns array of shape (nz, nx).
    """
    data = np.fromfile(filepath, dtype=np.float32, count=2 * nx * nz)
    if data.size < nx * nz:
        raise ValueError(
            f"Raw file too small for one image: {filepath} "
            f"(got {data.size}, need at least {nx * nz})"
        )

    # Backward compatibility: if only one image is present, treat it as total.
    if data.size < 2 * nx * nz:
        if signal_channel != "total":
            raise ValueError(
                f"Raw file does not contain primaries-only channel: {filepath}"
            )
        return data[: nx * nz].reshape(nz, nx)

    data = data.reshape(2, nz, nx)
    if signal_channel == "total":
        return data[0]
    if signal_channel == "primary":
        return data[1]
    raise ValueError(f"Unknown signal_channel='{signal_channel}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(in_root, results_dir, results_root, blank_results_root, signal_channel):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, results_dir)

    # Discover sweeps: {in_root}_xxxx.in, or fall back to {in_root}.in
    sweep_in_files = sorted(
        glob.glob(os.path.join(base_dir, f"{in_root}_*.in"))
    )

    single_mode = False
    if not sweep_in_files:
        single_file = os.path.join(base_dir, f"{in_root}.in")
        if os.path.exists(single_file):
            sweep_in_files = [single_file]
            single_mode = True
            print(f"Found single input file: {in_root}.in")
        else:
            raise FileNotFoundError(
                f"No {in_root}_*.in or {in_root}.in files found in {base_dir}"
            )
    else:
        print(f"Found {len(sweep_in_files)} sweep input files")

    all_rows = []

    for sweep_file in sweep_in_files:
        sweep_name = os.path.splitext(os.path.basename(sweep_file))[0]

        if single_mode:
            sweep_tag = ""
        else:
            sweep_id = sweep_name.replace(f"{in_root}_", "")
            sweep_tag = f"_{in_root}_{sweep_id}"

        print(f"\nProcessing {sweep_name}...")

        params = parse_in_file(sweep_file)

        # Find header files for this sweep (exclude .raw)
        pattern = os.path.join(results_dir, f"{results_root}{sweep_tag}_*")
        header_files = sorted(
            f for f in glob.glob(pattern) if not f.endswith(".raw")
        )

        if not header_files:
            print(f"  WARNING: no result files found for {sweep_name}, skipping")
            continue

        for header_file in header_files:
            info = parse_header(header_file)
            raw_file = header_file + ".raw"

            if not os.path.exists(raw_file):
                print(f"  WARNING: {raw_file} not found, skipping")
                continue

            nx, nz = info["nx"], info["nz"]
            n_pixels = nx * nz

            print(
                f"  Projection {info['proj_num']}/{info['total_proj']} "
                f"angle={info['angle_deg']:.1f}° "
                f"source=({info['source_pos'][0]:.1f}, "
                f"{info['source_pos'][1]:.1f}, "
                f"{info['source_pos'][2]:.1f})"
            )

            # Detector pixel positions -- shape (nz, nx, 3)
            det_pos = compute_detector_pixels(
                info["source_pos"],
                info["direction"],
                params["sdd"],
                params["width_x"],
                params["height_z"],
                nx, nz,
                params["offset_x"],
                params["offset_z"],
                params["rot_axis"],
            )

            # Pixel values -- shape (nz, nx)
            phantom_values = read_raw_projection(
                raw_file, nx, nz, signal_channel=signal_channel
            )

            # Find matching blank projection
            proj_suffix = os.path.basename(header_file).replace(
                f"{results_root}{sweep_tag}", ""
            )
            blank_header = os.path.join(
                results_dir, f"{blank_results_root}{sweep_tag}{proj_suffix}"
            )
            blank_raw = blank_header + ".raw"
            if not os.path.exists(blank_raw):
                raise FileNotFoundError(
                    f"Blank raw file not found: {blank_raw}"
                )
            blank_values = read_raw_projection(
                blank_raw, nx, nz, signal_channel=signal_channel
            )

            # Normalize: attenuation = ln(blank / phantom), clipped to >= 0
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.log(blank_values / phantom_values)
            values = np.clip(values, 0, None)

            # Flatten to (n_pixels, 3) and (n_pixels,)
            det_flat = det_pos.reshape(-1, 3)
            val_flat = values.reshape(-1)

            # Source is the same for every pixel in this projection
            src_repeated = np.broadcast_to(
                info["source_pos"], (n_pixels, 3)
            ).copy()

            # Build rows: [src_x, src_y, src_z, det_x, det_y, det_z, value]
            rows = np.empty((n_pixels, 7), dtype=np.float32)
            rows[:, 0:3] = src_repeated
            rows[:, 3:6] = det_flat
            rows[:, 6] = val_flat

            all_rows.append(rows)

    # Stack everything
    result = np.vstack(all_rows)
    print(f"\nTotal listmode entries: {result.shape[0]:,}")
    print(f"Columns: src_x, src_y, src_z, det_x, det_y, det_z, value")

    # Save
    out_path = os.path.join(base_dir, "listmode_data.npy")
    np.save(out_path, result)
    print(f"Saved to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build listmode data from MC-GPU simulation results."
    )
    parser.add_argument(
        "--in-root",
        required=True,
        help="Root name for input files (e.g. 'sweep' finds sweep_*.in)",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing the MC-GPU result files",
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Root name for result files (e.g. 'test_results' finds "
             "test_results_{sweep_tag}_*)",
    )
    parser.add_argument(
        "--blank-results-root",
        required=True,
        help="Root name for blank (air) result files (e.g. 'projection_blank' "
             "finds projection_blank_{sweep_tag}_*)",
    )
    parser.add_argument(
        "--signal-channel",
        choices=("total", "primary"),
        default="total",
        help="Signal channel from MC-GPU raw file: total=scatter+primary, primary=primary-only",
    )
    args = parser.parse_args()
    main(
        args.in_root,
        args.results_dir,
        args.results_root,
        args.blank_results_root,
        args.signal_channel,
    )
