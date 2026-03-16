"""Convert XCAT phantom attenuation data to Hounsfield Units and voxelId maps."""

import argparse
from pathlib import Path

import numpy as np

from utils import load_schneider_materials, parse_xcat_log


def main(input_path, log_path, output_path=None, output_voxel_path=None, n_partitions=3):
    input_path = Path(input_path)
    log_path = Path(log_path)
    output_path = Path(output_path) if output_path else input_path.with_name(input_path.stem + "_HU.bin")
    output_voxel_path = Path(output_voxel_path) if output_voxel_path else input_path.with_name(input_path.stem + "_voxelId.bin")

    # Read imaging parameters from XCAT log
    xcat = parse_xcat_log(log_path)

    X = xcat["array_size"]
    Y = xcat["array_size"]
    Z = xcat["n_slices"]
    pixel_width = xcat["pixel_width"]
    att_water = xcat["att_water"]

    print(f"Array: {X}x{Y}x{Z}, pixel_width={pixel_width} cm, att_water={att_water} 1/cm")

    raw = np.fromfile(input_path, dtype=np.float32)
    expected_size = X * Y * Z

    if raw.size != expected_size:
        raise ValueError(
            f"Unexpected file size. Read {raw.size} float32 values, "
            f"but expected {expected_size} for shape ({X}, {Y}, {Z})."
        )

    CT = raw.reshape((Z, Y, X))
    CT = CT / pixel_width  # Convert to cm^-1
    print(f"Input min/max: {CT.min():.6g} / {CT.max():.6g}")

    # To Hounsfield Units (HU)
    CT_HU = 1000 * ((CT - att_water) / att_water)

    print(f"HU min/max: {CT_HU.min():.6g} / {CT_HU.max():.6g}")

    # Map HU -> voxelId using same per-section partitioning as Generate_material_list
    schneider = load_schneider_materials()
    hu_sections = schneider["hu_material_sections"]
    n_sections = len(hu_sections) - 1

    # Assign voxelId by finding which subsection each HU falls into
    # Each section is split into n_partitions equal-width subsections
    voxel_ids = np.zeros(CT_HU.shape, dtype=np.int32)
    for section_idx in range(n_sections):
        hu_lo = float(hu_sections[section_idx])
        hu_hi = float(hu_sections[section_idx + 1])
        sub_edges = np.linspace(hu_lo, hu_hi, n_partitions + 1)
        for j in range(n_partitions):
            lo = sub_edges[j]
            hi = sub_edges[j + 1]
            if j < n_partitions - 1:
                mask = (CT_HU >= lo) & (CT_HU < hi)
            else:
                # Last subsection includes the upper boundary
                mask = (CT_HU >= lo) & (CT_HU <= hi)
            voxel_ids[mask] = section_idx * n_partitions + j

    voxel_ids = np.clip(voxel_ids, 0, 255).astype(np.uint8)

    print(f"VoxelId min/max: {voxel_ids.min()} / {voxel_ids.max()}")
    print(f"Unique voxelIds used: {len(np.unique(voxel_ids))}")

    # Save as uint8 binary (ready for MC-GPU)
    voxel_ids.tofile(output_voxel_path)
    print(f"Saved uint8 voxelId file to: {output_voxel_path}")

    # Also save the int16 HU file
    CT_HU_int16 = np.clip(np.rint(CT_HU), -32768, 32767).astype(np.int16)
    CT_HU_int16.tofile(output_path)
    print(f"Saved int16 HU file to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XCAT attenuation phantom to HU and voxelId maps.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the XCAT attenuation .bin file")
    parser.add_argument("--log", type=Path, required=True, help="Path to the XCAT log file")
    parser.add_argument("--output-hu", type=Path, default=None, help="Output path for int16 HU file (default: <input_dir>/<stem>_HU.bin)")
    parser.add_argument("--output-voxel", type=Path, default=None, help="Output path for uint8 voxelId file (default: <input_dir>/<stem>_voxelId.bin)")
    parser.add_argument("--n-partitions", type=int, default=3, help="Number of partitions per HU section (default: 3)")
    args = parser.parse_args()
    main(args.input, args.log, output_path=args.output_hu, output_voxel_path=args.output_voxel, n_partitions=args.n_partitions)
