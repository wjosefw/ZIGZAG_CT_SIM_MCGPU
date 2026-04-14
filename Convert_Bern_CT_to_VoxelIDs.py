"""Convert Bern CT .nii (attenuation map) to HU and voxelId maps using Bern material sections."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from Utils_Materials import load_bern_materials, convert_attenuation_map_to_hu


def main(input_path, output_path=None, output_voxel_path=None, output_blank_path=None, kVp=100):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(input_path.stem + "_HU.bin")
    output_voxel_path = Path(output_voxel_path) if output_voxel_path else input_path.with_name(input_path.stem + "_voxelId.bin")
    output_blank_path = Path(output_blank_path) if output_blank_path else input_path.with_name(input_path.stem + "_blank.bin")

    # Load .nii attenuation map
    nii = nib.load(str(input_path))
    # Shape convention: transpose to (Z, Y, X) matching the notebook
    att = nii.get_fdata()[:, :, :, 0].T if nii.ndim == 4 else nii.get_fdata().T
    att = att.astype(np.float32)
    print(f"Loaded {input_path.name}: shape={att.shape}, att range=[{att.min():.4g}, {att.max():.4g}]")

    # Convert attenuation -> HU
    CT_HU = convert_attenuation_map_to_hu(att, kVp=kVp)
    print(f"HU range: [{CT_HU.min():.4g}, {CT_HU.max():.4g}]")

    # Load Bern material sections
    bern = load_bern_materials()
    hu_sections = bern["hu_material_sections"]
    n_materials = len(hu_sections) - 1
    print(f"Bern materials: {n_materials}, HU sections: {hu_sections[0]:.4g} .. {hu_sections[-1]:.4g}")

    # Map HU -> voxelId (0-indexed: material 1 → id 0, material 2 → id 1, ...)
    voxel_ids = np.zeros(CT_HU.shape, dtype=np.int32)
    for i in range(n_materials):
        lo = float(hu_sections[i])
        hi = float(hu_sections[i + 1])
        if i < n_materials - 1:
            mask = (CT_HU >= lo) & (CT_HU < hi)
        else:
            mask = (CT_HU >= lo) & (CT_HU <= hi)
        voxel_ids[mask] = i

    voxel_ids = np.clip(voxel_ids, 0, 255).astype(np.uint8)
    print(f"VoxelId range: [{voxel_ids.min()}, {voxel_ids.max()}], unique: {len(np.unique(voxel_ids))}")

    # Save voxelId as uint8 binary (ready for MC-GPU)
    voxel_ids.tofile(output_voxel_path)
    print(f"Saved uint8 voxelId file to: {output_voxel_path}")

    # Save HU as int16 binary
    CT_HU_int16 = np.clip(np.rint(CT_HU), -32768, 32767).astype(np.int16)
    CT_HU_int16.tofile(output_path)
    print(f"Saved int16 HU file to: {output_path}")

    # Save blank (all-zero) phantom of same shape
    blank = np.zeros(CT_HU.shape, dtype=np.uint8)
    blank.tofile(output_blank_path)
    print(f"Saved blank (all-zero) phantom to: {output_blank_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Bern CT .nii attenuation map to HU and voxelId maps."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the .nii attenuation file")
    parser.add_argument("--output-hu", type=Path, default=None, help="Output path for int16 HU file")
    parser.add_argument("--output-voxel", type=Path, default=None, help="Output path for uint8 voxelId file")
    parser.add_argument("--output-blank", type=Path, default=None, help="Output path for blank (all-zero) phantom file")
    parser.add_argument("--kVp", type=int, default=100, help="CT tube voltage for attenuation→HU conversion (default: 100)")
    args = parser.parse_args()
    main(args.input, output_path=args.output_hu, output_voxel_path=args.output_voxel,
         output_blank_path=args.output_blank, kVp=args.kVp)
