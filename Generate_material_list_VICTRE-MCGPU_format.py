"""Generate material list file in MC-GPU format from Schneider compositions."""

import argparse

import numpy as np

from pathlib import Path
from utils import hu_to_density_schneider, load_schneider_materials


def main(material_dir="", n_bins=15):
    schneider = load_schneider_materials()

    print(f"Loaded {len(schneider['compositions_by_id'])} materials")
    print(f"Material HU sections: {schneider['hu_material_sections']}")

    # Build a voxelId -> (material_file, density) dictionary by splitting the full HU range into evenly spaced points
    hu_sections = schneider["hu_material_sections"]
    hu_start = int(hu_sections[0])      # -1000
    hu_end = int(hu_sections[-1]) - 1   #  2995 (last usable HU)
    bin_centers = np.linspace(hu_start, hu_end, n_bins)

    voxel_to_material = {}
    for voxel_id in range(n_bins):
        hu = bin_centers[voxel_id]
        # Find which Schneider section this HU belongs to
        section_idx = int(np.searchsorted(hu_sections, hu, side="right")) - 1
        material_id = section_idx + 1  # 1-indexed
        mat_prefix = f"{material_dir}/" if material_dir else ""
        material_file = f"{mat_prefix}MATERIAL{material_id}_5_150keV.mcgpu"
        density = float(hu_to_density_schneider(hu))
        voxel_to_material[voxel_id] = {
            "material_file": material_file,
            "density": density,
            "hu_center": float(hu),
        }

    print(f"Created {len(voxel_to_material)} entries (voxelId 0-{n_bins - 1})")
    print(f"HU step: {bin_centers[1] - bin_centers[0]:.2f}")
    print(f"\nExamples:")
    for vid in [0, 5, 10]:
        e = voxel_to_material[vid]
        print(
            f"  voxelId {vid:>3d} (HU~{e['hu_center']:>8.1f}) -> "
            f"{e['material_file']}, density={e['density']:.4f}"
        )

    # Write material list file in MC-GPU format
    material_list_path = Path("material_list.txt")

    with open(material_list_path, "w") as f:
        for voxel_id in range(n_bins):
            entry = voxel_to_material[voxel_id]
            mat_file = entry["material_file"]
            density = entry["density"]
            line_num = voxel_id + 1
            f.write(
                f"{mat_file:<45s} density={density:<10.4f} voxelId={voxel_id:<6d}"
                f"  # {line_num:>4d}th MATERIAL FILE\n"
            )

    print(f"\nWritten {n_bins} material entries to {material_list_path}")
    # Show first and last few lines
    lines = material_list_path.read_text().splitlines()
    print("\nFirst 5 lines:")
    for line in lines[:5]:
        print(line)
    print("\nLast 5 lines:")
    for line in lines[-5:]:
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="Optional directory prefix for material files (e.g. 'materials' -> 'materials/MATERIALx_...')",)
    parser.add_argument("--n-bins", type=int, default=15, help="Number of HU bins (default: 15)",)
    args = parser.parse_args()
    main(material_dir=args.prefix, n_bins=args.n_bins)
