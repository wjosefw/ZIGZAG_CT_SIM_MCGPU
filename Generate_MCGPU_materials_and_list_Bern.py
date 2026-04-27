"""Generate Bern .mcgpu files from .mat and write MC-GPU material list.

This script is independent from Schneider utilities and uses densities parsed
from Bern_materials.txt to build the material list.
"""

import argparse
from pathlib import Path

from Utils_Materials import load_bern_materials, run_mcgpu_material_creation, write_material_list_from_bern


def main(
    bern_path,
    mat_input_dir,
    mcgpu_output_dir,
    mcgpu_exe,
    material_prefix,
    e_min_keV=5,
    e_max_keV=150,
    n_points=1495,
):
    bern = load_bern_materials(bern_path)
    materials = bern["materials"]
    print(f"Loaded {len(materials)} Bern materials from {bern_path}")

    run_mcgpu_material_creation(
        mat_input_dir=mat_input_dir,
        mcgpu_output_dir=mcgpu_output_dir,
        mcgpu_exe=mcgpu_exe,
        e_min_keV=e_min_keV,
        e_max_keV=e_max_keV,
        n_points=n_points,
    )

    material_list_path = Path.cwd() / "material_list_bern.txt"
    write_material_list_from_bern(
        bern_materials=materials,
        mcgpu_output_dir=mcgpu_output_dir,
        material_list_path=material_list_path,
        material_prefix=material_prefix,
        e_min_keV=e_min_keV,
        e_max_keV=e_max_keV,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bern-path",
        required=True,
        help="Path to Bern material definition file",
    )
    parser.add_argument(
        "--mat-input-dir",
        "--PENELOPE-mat-dir",
        dest="mat_input_dir",
        required=True,
        help="Directory containing Bern .mat files",
    )
    parser.add_argument(
        "--mcgpu-output-dir",
        required=True,
        help="Directory to write Bern .mcgpu files",
    )
    parser.add_argument(
        "--mcgpu-exe",
        required=True,
        help="Path to MC-GPU_create_material_data.x executable",
    )
    parser.add_argument(
        "--material-prefix",
        default="",
        help=(
            "Prefix written in material list before each .mcgpu filename "
            "(default: basename of --mcgpu-output-dir)"
        ),
    )
    parser.add_argument(
        "--e-min-keV",
        type=int,
        default=5,
        help="Minimum energy in keV for MC-GPU material data (default: 5)",
    )
    parser.add_argument(
        "--e-max-keV",
        type=int,
        default=150,
        help="Maximum energy in keV for MC-GPU material data (default: 150)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=1495,
        help="Number of energy points for MC-GPU material data (default: 1495)",
    )
    args = parser.parse_args()
    main(
        bern_path=args.bern_path,
        mat_input_dir=args.mat_input_dir,
        mcgpu_output_dir=args.mcgpu_output_dir,
        mcgpu_exe=args.mcgpu_exe,
        material_prefix=args.material_prefix,
        e_min_keV=args.e_min_keV,
        e_max_keV=args.e_max_keV,
        n_points=args.n_points,
    )
