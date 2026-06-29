#!/usr/bin/env python3
"""
Build listmode data from MC-GPU simulation results.

For each detector pixel in each projection and sweep, outputs one row:
    source_x, source_y, source_z, det_x, det_y, det_z, pixel_value

Source positions and directions are parsed directly from MC-GPU output headers
so they match exactly what the simulation used.

Detector pixel world positions are computed from:
    det_center = source + direction * SDD
    pixel_world = det_center + local_x * X_det + local_z * Z_det

Where X_det and Z_det are the detector plane axes derived from the beam
direction and the rotation axis.

Outputs are saved as float32 numpy binary files (.npy):
  - listmode_data.npy: one row per detector pixel
  - projection_geometry.npy: one row per projection with source and detector center
"""

import numpy as np
import glob
import os
import argparse

import Sim_config
from Utils_Sim_Listmode import (
    parse_header,
    compute_detector_pixels,
    compute_detector_center,
    read_raw_both_channels,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_dir, results_root, blank_results_root, signal_channel):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, results_dir)

    # Discover all projection headers directly from results directory
    pattern = os.path.join(results_dir, f"{results_root}_*")
    header_files = sorted(f for f in glob.glob(pattern) if not f.endswith(".raw"))

    if not header_files:
        raise FileNotFoundError(
            f"No result files found matching {results_root}_* in {results_dir}"
        )
    print(f"Found {len(header_files)} projection headers")

    rot_axis = np.array([Sim_config.ROT_AXIS_X, Sim_config.ROT_AXIS_Y, Sim_config.ROT_AXIS_Z])

    all_rows = []
    projection_rows = []
    projection_images = []   # each entry: (nx, nz, 3): [total, primary, blank]
    projection_counter = 0

    for header_file in header_files:
        info = parse_header(header_file)
        raw_file = header_file + ".raw"

        if not os.path.exists(raw_file):
            print(f"  WARNING: {raw_file} not found, skipping")
            continue

        nx, nz = info["nx"], info["nz"]

        det_center = compute_detector_center(
            info["source_pos"], info["direction"], Sim_config.SDD
        )

        print(
            f"  Projection angle={info['angle_deg']:.1f}° "
            f"source=({info['source_pos'][0]:.1f}, "
            f"{info['source_pos'][1]:.1f}, "
            f"{info['source_pos'][2]:.1f})"
        )

        # Detector pixel positions -- shape (nz, nx, 3)
        det_pos = compute_detector_pixels(
            info["source_pos"],
            info["direction"],
            Sim_config.SDD,
            Sim_config.DETECTOR_WIDTH_X,
            Sim_config.DETECTOR_HEIGHT_Z,
            nx, nz,
            Sim_config.IMAGE_OFFSET_X,
            Sim_config.IMAGE_OFFSET_Z,
            rot_axis,
        )

        # Find matching blank projection by stripping results_root prefix
        proj_suffix = os.path.basename(header_file)[len(results_root):]
        blank_header = os.path.join(results_dir, f"{blank_results_root}{proj_suffix}")
        blank_raw = blank_header + ".raw"
        if not os.path.exists(blank_raw):
            raise FileNotFoundError(
                f"Blank raw file not found: {blank_raw}"
            )

        # Read both channels for phantom and blank -- shape (nx, nz, 2)
        phantom_both = read_raw_both_channels(raw_file, nx, nz)
        blank_both = read_raw_both_channels(blank_raw, nx, nz)

        # Normalize both channels: ln(blank / phantom), clipped to >= 0
        eps = 0.5
        log_both = np.log(np.maximum(blank_both, eps) / np.maximum(phantom_both, eps))
        log_both = np.clip(log_both, 0, None)   # shape (nx, nz, 2)

        # Dead pixels: both blank and phantom below half-photon floor -- exclude from listmode
        dead = (blank_both < eps) & (phantom_both < eps)   # (nx, nz, 2)

        # Select the channel used for listmode values.
        # log_both is (nx, nz, 2); det_pos is (nz, nx, 3) -- transpose needed.
        channel_idx = 0 if signal_channel == "total" else 1
        values = log_both[:, :, channel_idx].T   # (nz, nx)
        dead_mask = dead[:, :, channel_idx].T    # (nz, nx)

        # Flatten to (n_pixels, 3) and (n_pixels,)
        det_flat = det_pos.reshape(-1, 3)
        val_flat = values.reshape(-1)
        live = ~dead_mask.reshape(-1)

        det_flat = det_flat[live]
        val_flat = val_flat[live]

        n_live = live.sum()

        # Source is the same for every pixel in this projection
        src_repeated = np.broadcast_to(
            info["source_pos"], (n_live, 3)
        ).copy()

        # Build rows: [src_x, src_y, src_z, det_x, det_y, det_z, value]
        rows = np.empty((n_live, 7), dtype=np.float64)
        rows[:, 0:3] = src_repeated
        rows[:, 3:6] = det_flat
        rows[:, 6] = val_flat

        all_rows.append(rows)

        # Save raw counts: [total (scatter+primary), primary, blank] -- shape (nx, nz, 3)
        raw_counts = np.stack(
            [phantom_both[:, :, 0], phantom_both[:, :, 1], blank_both[:, :, 0]],
            axis=2,
        )
        projection_images.append(raw_counts)
        
        # Construct projection geometry object. 
        # u/v pixel-step vectors in world coordinates (same convention as
        # zigzag_3d ASTRA vectors: u = pix_size_x * x_det,
        #                          v = pix_size_z * z_det)
        x_det = np.cross(rot_axis, info["direction"])
        x_det /= np.linalg.norm(x_det)
        z_det = rot_axis.copy()
        pix_size_x = Sim_config.DETECTOR_WIDTH_X / nx
        pix_size_z = Sim_config.DETECTOR_HEIGHT_Z / nz
        u_world = pix_size_x * x_det
        v_world = pix_size_z * z_det

        # One explicit geometry row per projection:
        # [proj_idx, angle_deg, src_x, src_y, src_z,
        #  det_center_x, det_center_y, det_center_z,
        #  u_x, u_y, u_z, v_x, v_y, v_z]
        projection_rows.append(
            np.array(
                [
                    projection_counter,
                    info["angle_deg"],
                    info["source_pos"][0],
                    info["source_pos"][1],
                    info["source_pos"][2],
                    det_center[0],
                    det_center[1],
                    det_center[2],
                    u_world[0],
                    u_world[1],
                    u_world[2],
                    v_world[0],
                    v_world[1],
                    v_world[2],
                    info["direction"][0],
                    info["direction"][1],
                    info["direction"][2],
                ],
                dtype=np.float64,
            )
        )
        projection_counter += 1

    # Stack everything
    result = np.vstack(all_rows)
    print(f"\nTotal listmode entries: {result.shape[0]:,}")
    print(f"Columns: src_x, src_y, src_z, det_x, det_y, det_z, value")

    projection_geometry = np.vstack(projection_rows)
    print(f"Projection columns: proj_idx, angle_deg, "
        "src_x, src_y, src_z, det_center_x, det_center_y, det_center_z, "
        "u_x, u_y, u_z, v_x, v_y, v_z, dir_x, dir_y, dir_z"
    )
    
    # Save
    out_path = os.path.join(base_dir, "listmode_data.npy")
    np.save(out_path, result)
    print(f"Saved to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


    projection_out_path = os.path.join(base_dir, "projection_geometry.npy")
    np.save(projection_out_path, projection_geometry)
    print(f"Saved to {projection_out_path} "
          f"({os.path.getsize(projection_out_path) / 1e6:.1f} MB)"
    )

    # Stack raw images: list of N x (nx, nz, 2) -> (nx, nz, N, 2)
    images_array = np.stack(projection_images, axis=2)
    images_out_path = os.path.join(base_dir, "projection_images.npy")
    np.save(images_out_path, images_array)
    print(f"Saved projection images {images_array.shape} to {images_out_path} "
          f"({os.path.getsize(images_out_path) / 1e6:.1f} MB) "
          f"[channels: total, primary, blank]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build listmode data from MC-GPU simulation results."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing the MC-GPU result files",
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Root name for result files (e.g. 'proj_phantom' finds proj_phantom_*)",
    )
    parser.add_argument(
        "--blank-results-root",
        required=True,
        help="Root name for blank (air) result files (e.g. 'proj_blank' finds proj_blank_*)",
    )
    parser.add_argument(
        "--signal-channel",
        choices=("total", "primary"),
        default="total",
        help="Signal channel from MC-GPU raw file: total=scatter+primary, primary=primary-only",
    )
    args = parser.parse_args()
    main(
        args.results_dir,
        args.results_root,
        args.blank_results_root,
        args.signal_channel,
    )
