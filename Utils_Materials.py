# Utils_Materials.py
# Centralized utility functions for material handling

import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

def read_nii(file_path):
    """Load a NIfTI file and return (img, (zax, yax, xax)) with axes in mm."""
    nii = nib.load(file_path)
    img = nii.get_fdata()[:, :, :, 0].T if nii.ndim == 4 else nii.get_fdata().T
    img = img.astype(np.float32)
    imgdimz, imgdimy, imgdimx = img.shape
    voxdimx, voxdimy, voxdimz = np.abs(nii.affine).diagonal()[:3]
    zax = np.arange(-imgdimz*voxdimz/2, imgdimz*voxdimz/2 + voxdimz, voxdimz)
    yax = np.arange(-imgdimy*voxdimy/2, imgdimy*voxdimy/2 + voxdimy, voxdimy)
    xax = np.arange(-imgdimx*voxdimx/2, imgdimx*voxdimx/2 + voxdimx, voxdimx)
    return img, (zax, yax, xax)

# ---------------------------------------------------------------------------

def run_mcgpu_material_creation(mat_input_dir, mcgpu_output_dir, mcgpu_exe,
                                e_min_keV=5, e_max_keV=150, n_points=1495):
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
        out_file = mcgpu_output_dir / f"{mat_file.stem}_{e_min_keV}_{e_max_keV}keV.mcgpu"
        print(f"Processing {mat_file} -> {out_file} ...")
        # Keep paths short to avoid truncation in MC-GPU interactive input parser.
        mat_rel = os.path.relpath(mat_file, start=mcgpu_output_dir)
        out_name = out_file.name
        mcgpu_input = f"{e_min_keV * 1000},{e_max_keV * 1000}\n{n_points}\n{mat_rel}\n{out_name}\n"
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

# ---------------------------------------------------------------------------

def write_material_list(bern_materials, mcgpu_output_dir, material_list_path, material_prefix,
                                   e_min_keV=5, e_max_keV=150):
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
        filename = f"{name}_{e_min_keV}_{e_max_keV}keV.mcgpu"
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

# ---------------------------------------------------------------------------

def convert_attenuation_map_to_hu(attenuation_map, kVp=100):
    """Convert attenuation map at 511 keV to HU using the bilinear transformation.

    From https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.2174132
    """
    kVp_params = {
        80:  {"a": 3.64e-5, "b": 6.26e-2, "BP": 50},
        100: {"a": 4.43e-5, "b": 5.44e-2, "BP": 52},
        110: {"a": 4.92e-5, "b": 4.88e-2, "BP": 43},
        120: {"a": 5.10e-5, "b": 4.71e-2, "BP": 47},
        130: {"a": 5.51e-5, "b": 4.24e-2, "BP": 37},
        140: {"a": 5.64e-5, "b": 4.08e-2, "BP": 30},
    }
    if kVp not in kVp_params:
        raise ValueError(f"Unsupported kVp {kVp}. Supported: {sorted(kVp_params)}")
    p = kVp_params[kVp]
    breakpoint_att = 9.6e-5 * (p["BP"] + 1000)
    hu = np.where(
        attenuation_map <= breakpoint_att,
        attenuation_map / 9.6e-5 - 1000,
        (attenuation_map - p["b"]) / p["a"] - 1000,
    )
    return hu.astype(np.float32)

# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------

def write_named_material_input(output_dir, materials):
    """Write one .in file per material, using material names for file and .mat names."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for material in materials:
        name = material["name"]
        elements = material["elements"]
        density = material["density_g_cm3"]

        lines = []
        lines.append("1")          # Enter from keyboard
        lines.append(name)         # Material name
        n = len(elements)
        lines.append(str(n))       # Number of elements

        if n == 1:
            # Pure element: no weight fraction prompt
            lines.append(str(elements[0]["Z"]))
        else:
            # Compound: fraction by weight
            lines.append("2")
            for entry in elements:
                lines.append(str(entry["Z"]))
                lines.append(f"{entry['weight_fraction']:.6f}")

        lines.append("2")                  # Don't change I
        lines.append(str(density))         # Density in g/cm3
        lines.append("2")                  # Don't change Fcb/Wcb
        lines.append(f"{name}.mat")        # Output filename

        filepath = out / f"{name}.in"
        filepath.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(materials)} named .in files to {output_dir}/")
