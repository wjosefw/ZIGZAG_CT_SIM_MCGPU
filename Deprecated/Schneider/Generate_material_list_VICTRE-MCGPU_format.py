"""Generate material list file in MC-GPU format from Schneider compositions."""

import argparse

import numpy as np

from pathlib import Path
from Deprecated.Schneider.utils import hu_to_density_schneider, load_schneider_materials


def main(material_dir="", n_partitions=3):
    schneider = load_schneider_materials()

    print(f"Loaded {len(schneider['compositions_by_id'])} materials")
    print(f"Material HU sections: {schneider['hu_material_sections']}")

    # Build a voxelId -> (material_file, density) dictionary by partitioning
    # each HU section into n_partitions representative points
    hu_sections = schneider["hu_material_sections"]
    n_sections = len(hu_sections) - 1

    voxel_to_material = {}
    voxel_id = 0
    for section_idx in range(n_sections):
        hu_lo = float(hu_sections[section_idx])
        hu_hi = float(hu_sections[section_idx + 1])
        material_id = section_idx + 1  # 1-indexed
        mat_prefix = f"{material_dir}/" if material_dir else ""
        material_file = f"{mat_prefix}MATERIAL{material_id}_5_150keV.mcgpu"

        # Split section into n_partitions equal-width subsections, use midpoints
        sub_edges = np.linspace(hu_lo, hu_hi, n_partitions + 1)
        for j in range(n_partitions):
            hu = (sub_edges[j] + sub_edges[j + 1]) / 2.0
            density = float(hu_to_density_schneider(hu))
            voxel_to_material[voxel_id] = {
                "material_file": material_file,
                "density": density,
                "hu_center": float(hu),
            }
            voxel_id += 1

    n_total = len(voxel_to_material)
    print(f"Created {n_total} entries (voxelId 0-{n_total - 1})")
    print(f"{n_sections} HU sections x {n_partitions} partitions each")
    print(f"\nExamples:")
    for vid in [0, n_total // 2, n_total - 1]:
        e = voxel_to_material[vid]
        print(
            f"  voxelId {vid:>3d} (HU~{e['hu_center']:>8.1f}) -> "
            f"{e['material_file']}, density={e['density']:.4f}"
        )

    # Write material list file in MC-GPU format
    material_list_path = Path("material_list.txt")

    with open(material_list_path, "w") as f:
        for voxel_id in range(n_total):
            entry = voxel_to_material[voxel_id]
            mat_file = entry["material_file"]
            density = entry["density"]
            line_num = voxel_id + 1
            f.write(
                f"{mat_file:<45s} density={density:<10.4f} voxelId={voxel_id:<6d}"
                f"  # {line_num:>4d}th MATERIAL FILE\n"
            )

    print(f"\nWritten {n_total} material entries to {material_list_path}")
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
    parser.add_argument("--n-partitions", type=int, default=3, help="Number of partitions per HU section (default: 3)",)
    args = parser.parse_args()
    main(material_dir=args.prefix, n_partitions=args.n_partitions)
