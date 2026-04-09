from __future__ import annotations

import argparse
import numpy as np
import array_api_compat.cupy as xp
# import array_api_compat.numpy as xp
# import array_api_compat.torch as xp

from array_api_compat import to_device
from utils import sirt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listmode_file", required=True, help="Path to listmode .npy file")
    parser.add_argument("--img_size",    required=True, nargs=3, type=int,   metavar=("N0", "N1", "N2"), help="Image dimensions in voxels")
    parser.add_argument("--voxel_size",  required=True, nargs=3, type=float, metavar=("DX", "DY", "DZ"), help="Voxel size in cm")
    parser.add_argument("--img_origin",  required=True, nargs=3, type=float, metavar=("X0", "Y0", "Z0"), help="World coordinate of center of voxel [0,0,0] in cm")
    parser.add_argument("--num_iter", type=int, default=50, help="Number of reconstruction iterations")
    parser.add_argument("--relaxation", type=float, default=1.0, help="SIRT relaxation factor")
    parser.add_argument("--relres_tol", type=float, default=None, help="Early stop SIRT when ||Ax-b||/||b|| <= tol")
    args = parser.parse_args()

    if "numpy" in xp.__name__:
        dev = "cpu"
    elif "cupy" in xp.__name__:
        dev = xp.cuda.Device(0)
    elif "torch" in xp.__name__:
        dev = "cuda"

    # ── Image / geometry parameters ──────────────────────────────────────────────
    n0, n1, n2 = args.img_size
    img_dim = (n0, n1, n2)
    voxel_size = to_device(xp.asarray(args.voxel_size, dtype=xp.float32), dev)
    img_origin = to_device(xp.asarray(args.img_origin, dtype=xp.float32), dev)

    # ── Data loading ─────────────────────────────────────────────────────────────
    listmode_data = np.load(args.listmode_file)
    xstart = to_device(xp.asarray(listmode_data[:, :3], dtype=xp.float32), dev)
    xend   = to_device(xp.asarray(listmode_data[:, 3:6], dtype=xp.float32), dev)
    y      = to_device(xp.asarray(listmode_data[:, 6],   dtype=xp.float32), dev)

    # ── Reconstruction from simulation data ──────────────────────────────────────
    x = sirt(
        xstart, xend, y, img_dim, img_origin, voxel_size,
        num_iter=args.num_iter,
        relaxation=args.relaxation,
        monitor_every=5,
        relres_tol=args.relres_tol,
    )

    # ── Save reconstruction as raw ────────────────────────────────────────────────
    # Transposes are just a convention: parallelproj uses (x,y,z), phantoms are stored as (z,y,x)
    x_np = np.transpose(np.asarray(x.get(), dtype=np.float32), (2, 1, 0))
    x_np.tofile("recon.raw")
    print(f"Saved recon.raw  shape={x_np.shape}  dtype={x_np.dtype}")
