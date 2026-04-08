"""Generate Bern .mcgpu files from .mat and write MC-GPU material list.

This script is independent from Schneider utilities and uses densities parsed
from Bern_materials.txt to build the material list.
"""

import argparse
import os
import subprocess
from pathlib import Path

from utils import load_bern_materials


def run_mcgpu_material_creation(mat_input_dir, mcgpu_output_dir, mcgpu_exe):
    """Run MC-GPU_create_material_data.x for each Bern .mat file."""
    mat_input_dir = Path(mat_input_dir).resolve()
    mcgpu_output_dir = Path(mcgpu_output_dir).resolve()
    mcgpu_exe = Path(mcgpu_exe).resolve()

    if not mcgpu_exe.exists():
        raise FileNotFoundError(f"MC-GPU executable not found at: {mcgpu_exe}")

    mat_files = sorted(mat_input_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {mat_input_dir}")

    mcgpu_output_dir.mkdir(parents=True, exist_ok=True)

    for mat_file in mat_files:
        out_file = mcgpu_output_dir / f"{mat_file.stem}_5_150keV.mcgpu"
        print(f"Processing {mat_file} -> {out_file} ...")
        # Keep paths short to avoid truncation in MC-GPU interactive input parser.
        mat_rel = os.path.relpath(mat_file, start=mcgpu_output_dir)
        out_name = out_file.name
        mcgpu_input = f"5000,150000\n1495\n{mat_rel}\n{out_name}\n"
        subprocess.run(
            [str(mcgpu_exe)],
            input=mcgpu_input,
            text=True,
            check=True,
            cwd=str(mcgpu_output_dir),
        )
        if not out_file.exists():
            raise FileNotFoundError(
                f"MC-GPU did not create expected output file: {out_file}"
            )

    print(f"Created {len(mat_files)} .mcgpu files in {mcgpu_output_dir}")


def write_material_list_from_bern(bern_materials, mcgpu_output_dir, material_list_path, material_prefix):
    """Write material_list in MC-GPU format using Bern densities."""
    mcgpu_output_dir = Path(mcgpu_output_dir).resolve()
    material_list_path = Path(material_list_path).resolve()

    if material_prefix:
        prefix = material_prefix.rstrip("/")
    else:
        prefix = mcgpu_output_dir.name

    lines = []
    for voxel_id, material in enumerate(bern_materials):
        name = material["name"]
        density = material["density_g_cm3"]
        filename = f"{name}_5_150keV.mcgpu"
        full_file = mcgpu_output_dir / filename
        if not full_file.exists():
            raise FileNotFoundError(
                f"Expected .mcgpu file missing for material '{name}': {full_file}"
            )
        material_file = f"{prefix}/{filename}" if prefix else filename
        line_num = voxel_id + 1
        lines.append(
            f"{material_file:<45s} density={density:<10.4f} voxelId={voxel_id:<6d}"
            f"  # {line_num:>4d}th MATERIAL FILE"
        )

    material_list_path.write_text("\n".join(lines) + "\n")
    print(f"Written {len(lines)} entries to {material_list_path}")


def main(
    bern_path,
    mat_input_dir,
    mcgpu_output_dir,
    mcgpu_exe,
    material_prefix,
):
    bern = load_bern_materials(bern_path)
    materials = bern["materials"]
    print(f"Loaded {len(materials)} Bern materials from {bern_path}")

    run_mcgpu_material_creation(
        mat_input_dir=mat_input_dir,
        mcgpu_output_dir=mcgpu_output_dir,
        mcgpu_exe=mcgpu_exe,
    )

    material_list_path = Path.cwd() / "material_list_bern.txt"
    write_material_list_from_bern(
        bern_materials=materials,
        mcgpu_output_dir=mcgpu_output_dir,
        material_list_path=material_list_path,
        material_prefix=material_prefix,
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
    args = parser.parse_args()
    main(
        bern_path=args.bern_path,
        mat_input_dir=args.mat_input_dir,
        mcgpu_output_dir=args.mcgpu_output_dir,
        mcgpu_exe=args.mcgpu_exe,
        material_prefix=args.material_prefix,
    )
