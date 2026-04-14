import parallelproj
from array_api_compat import get_namespace, device, to_device


def sirt(
    xstart,
    xend,
    measurements,
    img_dim,
    img_origin,
    voxel_size,
    num_iter=50,
    relaxation=1.0,
    nonnegativity=True,
    monitor_every=5,
    relres_tol=None,
):
    """SIRT reconstruction from listmode ray data.

    Uses normalized updates:
        x_{k+1} = x_k + relaxation * C * A^T * (R * (b - A x_k))
    where
        R_i = 1 / (sum_j A_ij)
        C_j = 1 / (sum_i A_ij)
    Args:
        monitor_every: Print progress every N iterations.
        relres_tol: If set, stop early when ||Ax-b|| / ||b|| <= relres_tol.
    """
    xp = get_namespace(measurements)
    dev = device(xstart)

    ones_meas = xp.ones(measurements.shape, dtype=xp.float32)
    ones_img = xp.ones(img_dim, dtype=xp.float32)

    # Row normalization (per-ray path length sum)
    row_sum = parallelproj.joseph3d_fwd(xstart, xend, ones_img, img_origin, voxel_size)
    row_sum = xp.maximum(row_sum, xp.asarray(1e-6, dtype=xp.float32))

    # Column normalization (voxel sensitivity)
    col_sum = parallelproj.joseph3d_back(
        xstart, xend, img_dim, img_origin, voxel_size, ones_meas
    )
    col_sum = xp.maximum(col_sum, xp.asarray(1e-6, dtype=xp.float32))

    x = to_device(xp.zeros(img_dim, dtype=xp.float32), dev)
    relaxation = xp.asarray(relaxation, dtype=xp.float32)
    monitor_every = max(1, int(monitor_every))
    meas_norm = xp.sqrt(xp.sum(measurements * measurements))
    meas_norm = xp.maximum(meas_norm, xp.asarray(1e-12, dtype=xp.float32))
    last_relres = None

    for i in range(num_iter):
        proj = parallelproj.joseph3d_fwd(xstart, xend, x, img_origin, voxel_size)
        residual = measurements - proj
        weighted_residual = residual / row_sum
        backproj = parallelproj.joseph3d_back(
            xstart, xend, img_dim, img_origin, voxel_size, weighted_residual
        )
        x = x + relaxation * (backproj / col_sum)
        if nonnegativity:
            x = xp.maximum(x, xp.asarray(0, dtype=xp.float32))
        relres = xp.sqrt(xp.sum(residual * residual)) / meas_norm
        last_relres = float(relres)

        iter_idx = i + 1
        if (iter_idx % monitor_every == 0) or (iter_idx == 1) or (iter_idx == num_iter):
            print(
                f"SIRT iteration {iter_idx:03} / {num_iter:03}  "
                f"relres={last_relres:.6e}  max={float(x.max()):.4f}"
            )

        if (relres_tol is not None) and (last_relres <= relres_tol):
            print(
                f"Early stopping at iteration {iter_idx:03}: "
                f"relres={last_relres:.6e} <= tol={relres_tol:.6e}"
            )
            break

    if last_relres is None:
        last_relres = float("nan")
    print(f"Done. Recon max: {float(x.max()):.4f}, final relres={last_relres:.6e}")
    return x
