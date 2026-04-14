"""Generate .in and .mat files for PENELOPE material.x from Bern definitions.

This script is independent from the Schneider pipeline and preserves material names
from Bern_materials.txt in both .in and .mat file names.
"""

import argparse

from Utils_Materials import load_bern_materials, run_material_creation, write_named_material_input


def main(
    output_dir,
    bern_path,
    mat_output_dir,
    pendbase_dir,
):
    bern = load_bern_materials(bern_path)
    materials = bern["materials"]
    print(f"Loaded {len(materials)} Bern materials from {bern_path}")

    if materials:
        sample = materials[0]
        print(
            f"Example: {sample['name']} "
            f"(density={sample['density_g_cm3']:.6f} g/cm3, "
            f"elements={len(sample['elements'])})"
        )

    write_named_material_input(output_dir=output_dir, materials=materials)

    run_material_creation(
        input_dir=output_dir,
        output_dir=mat_output_dir,
        pendbase_dir=pendbase_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write Bern-named .in files to",
    )
    parser.add_argument(
        "--bern-path",
        required=True,
        help="Path to Bern material definition file",
    )
    parser.add_argument(
        "--mat-output-dir",
        required=True,
        help="Directory to write .mat files when running material.x",
    )
    parser.add_argument(
        "--pendbase-dir",
        required=True,
        help="Directory containing material.x and PENELOPE data files",
    )
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        bern_path=args.bern_path,
        mat_output_dir=args.mat_output_dir,
        pendbase_dir=args.pendbase_dir,
    )
