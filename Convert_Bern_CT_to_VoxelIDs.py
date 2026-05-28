"""Convert Bern CT .nii (attenuation map) to HU and voxelId maps using Bern material sections."""

import argparse
from pathlib import Path

import numpy as np

from bern_materials_db import HU_SECTIONS
from Utils_Materials import convert_attenuation_map_to_hu, read_nii


def main(input_path, kVp=100):
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_HU.bin")
    output_voxel_path = input_path.with_name(input_path.stem + "_voxelId.bin")
    output_blank_path = input_path.with_name(input_path.stem + "_blank.bin")

    # Load .nii attenuation map
    att, (zax, yax, xax) = read_nii(input_path)
    imgdimz, imgdimy, imgdimx = att.shape
    voxdimz = zax[1] - zax[0]
    voxdimy = yax[1] - yax[0]
    voxdimx = xax[1] - xax[0]
    
    print(f"Loaded: {input_path.name}")
    print(f"Att range: [{att.min():.4g}, {att.max():.4g}]")
    print(f'Image Dimensions: {imgdimz, imgdimy, imgdimx}')
    print(f'Voxel dimensions: {voxdimz, voxdimy, voxdimx}')
    print(f'Image Center: {imgdimz*voxdimz/2:.3f}, {imgdimy*voxdimy/2:.3f}, {imgdimx*voxdimx/2:.3f}')
    print(f'Center Position of voxel (0,0,0) if phantom is centered in coordinate origin: {(-imgdimz*voxdimz/2) + voxdimz/2:.5f}, {(-imgdimy*voxdimy/2) + voxdimy/2:.5f}, {(-imgdimy*voxdimy/2) + voxdimy/2:.5f}')

    # Convert attenuation -> HU
    CT_HU = convert_attenuation_map_to_hu(att, kVp=kVp)
    print(f"HU range: [{CT_HU.min():.4g}, {CT_HU.max():.4g}]")

    n_materials = len(HU_SECTIONS) - 1
    print(f"Bern materials: {n_materials}, HU sections: {HU_SECTIONS[0]:.4g} .. {HU_SECTIONS[-1]:.4g}")

    # Map HU -> voxelId (0-indexed: material 1 → id 0, material 2 → id 1, ...)
    voxel_ids = np.zeros(CT_HU.shape, dtype=np.int32)
    for i in range(n_materials):
        lo = float(HU_SECTIONS[i])
        hi = float(HU_SECTIONS[i + 1])
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
    parser.add_argument("--kVp", type=int, default=100, help="CT tube voltage for attenuation→HU conversion (default: 100)")
    args = parser.parse_args()
    main(args.input, kVp=args.kVp)
