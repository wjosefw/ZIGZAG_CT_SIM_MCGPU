"""Generate .in and .mat files for PENELOPE material.x from Bern definitions.

This script is independent from the Schneider pipeline and preserves material names
from Bern_materials.txt in both .in and .mat file names.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from utils import load_bern_materials, write_named_material_input


def run_material_creation(input_dir, output_dir, pendbase_dir):
    """Run material.x for each Bern .in file and move resulting .mat files."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    pendbase_dir = Path(pendbase_dir).resolve()
    material_x = pendbase_dir / "material.x"

    if not material_x.exists():
        raise FileNotFoundError(f"material.x not found at: {material_x}")

    in_files = sorted(input_dir.glob("*.in"))
    if not in_files:
        raise FileNotFoundError(f"No .in files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for in_file in in_files:
        mat_name = f"{in_file.stem}.mat"
        print(f"Processing {in_file} -> {output_dir / mat_name} ...")
        with in_file.open("r") as f_in:
            subprocess.run(
                [str(material_x)],
                stdin=f_in,
                cwd=str(pendbase_dir),
                check=True,
            )

        created_mat = pendbase_dir / mat_name
        if not created_mat.exists():
            raise FileNotFoundError(
                f"Expected output file not created: {created_mat} "
                f"(input file: {in_file})"
            )

        shutil.move(str(created_mat), str(output_dir / mat_name))

    print(f"Created {len(in_files)} .mat files in {output_dir}")


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
