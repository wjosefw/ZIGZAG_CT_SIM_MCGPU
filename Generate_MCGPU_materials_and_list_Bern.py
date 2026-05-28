"""Generate .mcgpu cross-section files and a MC-GPU material list.

Material data is imported directly from bern_materials_db or gate_materials_db
— no text-file parsing at runtime.

Prerequisites
-------------
PENELOPE .mat files must already exist — run Generate_PENELOPE_materials_Bern.py
first.

Usage example
-------------
python Generate_MCGPU_materials_and_list_Bern.py --source bern \\
    --PENELOPE-mat-dir pendbase/ \\
    --mcgpu-output-dir MC_GPU_materials_bern/ \\
    --mcgpu-exe        MC-GPU_create_material_data.x

python Generate_MCGPU_materials_and_list_Bern.py --source gate \\
    --PENELOPE-mat-dir pendbase/ \\
    --mcgpu-output-dir MC_GPU_materials_gate/ \\
    --mcgpu-exe        MC-GPU_create_material_data.x
"""

import argparse
from pathlib import Path

from bern_materials_db import MATERIALS as BERN_MATERIALS
from gate_materials_db import MATERIALS as GATE_MATERIALS
from Utils_Materials import run_mcgpu_material_creation, write_material_list

SOURCES = {
    "bern": BERN_MATERIALS,
    "gate": GATE_MATERIALS,
}


def main(
    source,
    mat_input_dir,
    mcgpu_output_dir,
    mcgpu_exe,
    e_min_keV=5,
    e_max_keV=150,
    n_points=1495,
):
    materials = SOURCES[source]
    print(f"Loaded {len(materials)} materials from {source}_materials_db")

    run_mcgpu_material_creation(
        mat_input_dir=mat_input_dir,
        mcgpu_output_dir=mcgpu_output_dir,
        mcgpu_exe=mcgpu_exe,
        e_min_keV=e_min_keV,
        e_max_keV=e_max_keV,
        n_points=n_points,
    )

    material_list_path = Path.cwd() / f"material_list_{source}.txt"
    write_material_list(
        materials,
        mcgpu_output_dir=mcgpu_output_dir,
        material_list_path=material_list_path,
        material_prefix="",
        e_min_keV=e_min_keV,
        e_max_keV=e_max_keV,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["bern", "gate"],
        default="bern",
        help="Material database to use (default: bern)",
    )
    parser.add_argument(
        "--PENELOPE-mat-dir",
        dest="mat_input_dir",
        required=True,
        help="Directory containing the PENELOPE .mat files",
    )
    parser.add_argument(
        "--mcgpu-output-dir",
        required=True,
        help="Directory to write .mcgpu cross-section files",
    )
    parser.add_argument(
        "--mcgpu-exe",
        required=True,
        help="Path to MC-GPU_create_material_data.x executable",
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
        source=args.source,
        mat_input_dir=args.mat_input_dir,
        mcgpu_output_dir=args.mcgpu_output_dir,
        mcgpu_exe=args.mcgpu_exe,
        e_min_keV=args.e_min_keV,
        e_max_keV=args.e_max_keV,
        n_points=args.n_points,
    )
