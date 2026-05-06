#------------------------------------------------------
#----------------- List mode helpers ------------------
#------------------------------------------------------


import os
import re
import subprocess
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_header(filepath):
    """Parse an MC-GPU output header file for per-projection geometry."""
    with open(filepath, "r") as f:
        text = f.read()

    m = re.search(
        r"projection\s+\d+\s+of\s+\d+.*?angle\s*=\s*([\d.]+)", text
    )
    angle_deg = float(m.group(1))

    m = re.search(
        r"Focal spot position = \(([-\d.]+),([-\d.]+),([-\d.]+)\).*?"
        r"direction = \(([-\d.]+),([-\d.]+),([-\d.]+)\)",
        text,
    )
    source_pos = np.array([float(m.group(i)) for i in range(1, 4)])
    direction = np.array([float(m.group(i)) for i in range(4, 7)])

    m = re.search(r"Number of pixels in X and Z:\s+(\d+)\s+(\d+)", text)
    nx, nz = int(m.group(1)), int(m.group(2))

    return {
        "angle_deg": angle_deg,
        "source_pos": source_pos,
        "direction": direction,
        "nx": nx,
        "nz": nz,
    }


def parse_in_file(filepath):
    """Parse an MC-GPU .in file for detector geometry parameters."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    def first_token(line):
        return line.split("#")[0].strip().split()

    # Line 41: IMAGE SIZE (width, height) Dx Dz [cm]
    tokens = first_token(lines[40])
    width_x, height_z = float(tokens[0]), float(tokens[1])

    # Line 42: SOURCE-TO-DETECTOR DISTANCE
    tokens = first_token(lines[41])
    sdd = float(tokens[0])

    # Line 43: IMAGE OFFSET
    tokens = first_token(lines[42])
    offset_x, offset_z = float(tokens[0]), float(tokens[1])

    # Line 56: SOURCE-TO-ROTATION AXIS DISTANCE (= SOD, valid when rotation axis is at origin)
    tokens = first_token(lines[55])
    sod = float(tokens[0])

    # Line 59: AXIS OF ROTATION
    tokens = first_token(lines[58])
    rot_axis = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])])
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    return {
        "sdd": sdd,
        "sod": sod,
        "width_x": width_x,
        "height_z": height_z,
        "offset_x": offset_x,
        "offset_z": offset_z,
        "rot_axis": rot_axis,
    }


# ---------------------------------------------------------------------------
# Detector geometry
# ---------------------------------------------------------------------------

def compute_detector_pixels(source_pos, direction, sdd, width_x, height_z,
                            nx, nz, offset_x, offset_z, rot_axis):
    """
    Compute world coordinates of every detector pixel center.

    Returns array of shape (nz, nx, 3) matching the MC-GPU raw file layout
    (Z-slow, X-fast).
    """
    det_center = source_pos + direction * sdd

    # Detector plane axes
    # Using cross(rot_axis, direction) to match ASTRA's pixel-0 convention (-X side).
    # Raw X read order is also flipped in read_raw_both_channels to keep
    # each measurement assigned to the correct world position.
    x_det = np.cross(rot_axis, direction)
    x_det /= np.linalg.norm(x_det)
    z_det = rot_axis.copy()

    # Local pixel coordinates (center of each pixel)
    pix_size_x = width_x / nx
    pix_size_z = height_z / nz

    local_x = (np.arange(nx) + 0.5) * pix_size_x - 0.5 * width_x + offset_x
    local_z = (np.arange(nz) + 0.5) * pix_size_z - 0.5 * height_z + offset_z

    # Broadcast to (nz, nx, 3)
    # ii shape (nz, nx), jj shape (nz, nx)
    ii, jj = np.meshgrid(local_x, local_z)

    positions = (
        det_center[np.newaxis, np.newaxis, :]
        + ii[:, :, np.newaxis] * x_det[np.newaxis, np.newaxis, :]
        + jj[:, :, np.newaxis] * z_det[np.newaxis, np.newaxis, :]
    )
    return positions  # (nz, nx, 3)


def compute_detector_center(source_pos, direction, sdd):
    """Compute detector center world coordinates for one projection."""
    return source_pos + direction * sdd


# ---------------------------------------------------------------------------
# Raw file reading
# ---------------------------------------------------------------------------


def read_raw_both_channels(filepath, nx, nz):
    """
    Read both channels from an MC-GPU .raw file.

    Returns array of shape (nx, nz, 2):
      - [..., 0]: total signal (scatter + primaries)
      - [..., 1]: primaries only
    """
    data = np.fromfile(filepath, dtype=np.float32, count=2 * nx * nz)
    if data.size < 2 * nx * nz:
        raise ValueError(
            f"Raw file does not contain both channels: {filepath} "
            f"(got {data.size}, need {2 * nx * nz})"
        )
    data = data.reshape(2, nz, nx)
    # Flip X axis to match ASTRA pixel-0 convention (paired with x_det sign flip
    # in compute_detector_pixels so each measurement stays at the correct world position).
    data = data[:, :, ::-1]
    # Transpose from (2, nz, nx) -> (nx, nz, 2)
    return data.transpose(2, 1, 0)


#------------------------------------------------------
#-------------- Simulation helpers --------------------
#------------------------------------------------------

import re


def parse_template(filepath):
    """Read the template .in file and return its lines."""
    with open(filepath) as f:
        return f.readlines()


def find_and_replace_value(lines, comment_keyword, new_value_str):
    """Find a line by its trailing comment keyword and replace the leading value."""
    for i, line in enumerate(lines):
        if comment_keyword in line and not line.strip().startswith('#'):
            match = re.match(r'^(\s*)(.*?)(#.*)', line)
            if match:
                indent, _old_val, comment = match.groups()
                lines[i] = f"{indent}{new_value_str}   {comment}\n"
            return lines
    raise ValueError(f"Could not find line with comment keyword: {comment_keyword}")


def update_source_z(lines, new_z):
    """Update the Z component of SOURCE POSITION."""
    for i, line in enumerate(lines):
        if 'SOURCE POSITION' in line and not line.strip().startswith('#'):
            match = re.match(r'^(\s*)([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)(.*)', line)
            if match:
                indent, x, y, _z, rest = match.groups()
                lines[i] = f"{indent}{x}  {y}   {new_z:.6f}{rest}\n"
            return lines
    raise ValueError("Could not find SOURCE POSITION line")


def compute_euler_angles(dir_x, dir_y, dir_z):
    """
    RzRyRz Euler angles that rotate the MC-GPU default direction (0,1,0) to
    (dir_x, dir_y, dir_z). Valid when dir_z ≈ 0 (source-detector in XY plane).
    Returns (alpha, beta, gamma) in degrees.
    """
    alpha = np.degrees(np.arctan2(dir_x, -dir_y))
    beta  = 0.0
    gamma = 180.0
    return alpha, beta, gamma


def update_euler_angles(lines, alpha, beta, gamma):
    """Update the RzRyRz Euler angles for the source beam orientation."""
    return find_and_replace_value(
        lines, 'EULER ANGLES', f"{alpha:.6f}   {beta:.6f}   {gamma:.6f}"
    )


def update_source_position(lines, x, y, z):
    """Update X, Y, Z of SOURCE POSITION."""
    return find_and_replace_value(lines, 'SOURCE POSITION', f"{x:.8f}  {y:.8f}  {z:.8f}")


def update_source_direction(lines, u, v, w):
    """Update direction cosines U V W."""
    return find_and_replace_value(lines, 'SOURCE DIRECTION COSINES', f"{u:.8f}  {v:.8f}  {w:.8f}")


def update_num_projections(lines, n_proj):
    return find_and_replace_value(lines, 'NUMBER OF PROJECTIONS', str(n_proj))


def update_angular_rotation_to_first(lines, angle_deg):
    return find_and_replace_value(lines, 'ANGULAR ROTATION TO FIRST PROJECTION', f"{angle_deg:.6f}")


def update_translation_along_axis(lines, translation_cm):
    return find_and_replace_value(lines, 'TRANSLATION ALONG ROTATION AXIS', f"{translation_cm:.6f}")


def update_output_name(lines, name):
    return find_and_replace_value(lines, 'OUTPUT IMAGE FILE NAME', name)


def update_angle_between_projections(lines, angle_deg):
    return find_and_replace_value(lines, 'ANGLE BETWEEN PROJECTIONS', f"{angle_deg:.6f}")


def update_phantom_file(lines, phantom_path):
    """Replace the voxelized geometry file path (first non-comment line after
    the VOXELIZED GEOMETRY FILE section header)."""
    in_section = False
    for i, line in enumerate(lines):
        if 'SECTION VOXELIZED GEOMETRY FILE' in line:
            in_section = True
            continue
        if in_section and line.strip() and not line.strip().startswith('#'):
            lines[i] = f"{phantom_path}\n"
            return lines
    raise ValueError("Could not find voxelized geometry file line")


def get_initial_z(lines):
    """Extract the initial Z coordinate from SOURCE POSITION."""
    for line in lines:
        if 'SOURCE POSITION' in line and not line.strip().startswith('#'):
            match = re.match(r'^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)', line)
            if match:
                return float(match.group(3))
    raise ValueError("Could not find SOURCE POSITION line")


def get_output_base_name(lines):
    """Extract the base output name from the template."""
    for line in lines:
        if 'OUTPUT IMAGE FILE NAME' in line and not line.strip().startswith('#'):
            match = re.match(r'^\s*(\S+)', line)
            if match:
                return match.group(1)
    raise ValueError("Could not find OUTPUT IMAGE FILE NAME line")


#------------------------------------------------------
#--------------- NIfTI / geometry helpers -------------
#------------------------------------------------------

import nibabel as nib


def read_nii(file_path):
    """Load a NIfTI file and return (img, (zax, yax, xax)) with axes in mm."""
    nii = nib.load(file_path)
    img = nii.get_fdata()[:, :, :, 0].T
    imgdimz, imgdimy, imgdimx = img.shape
    voxdimx, voxdimy, voxdimz = np.abs(nii.affine).diagonal()[:3]
    zax = np.arange(-imgdimz*voxdimz/2, imgdimz*voxdimz/2 + voxdimz, voxdimz)
    yax = np.arange(-imgdimy*voxdimy/2, imgdimy*voxdimy/2 + voxdimy, voxdimy)
    xax = np.arange(-imgdimx*voxdimx/2, imgdimx*voxdimx/2 + voxdimx, voxdimx)
    return img, (zax, yax, xax)


def subsample_slicewise(vectors, xax, yax, zmin=-100, zmax=100,
                         D_size=(1000, 50), D_z=1, SOD=650, ODD=450,
                         return_mask=False):
    """
    Filter rows of *vectors* whose Z coordinate (column 2) falls within
    [zmin, zmax] extended by the cone-beam margin needed to cover the detector.

    Parameters
    ----------
    vectors   : (N, ≥3) array — column 2 must be source Z [cm]
    xax, yax  : phantom spatial axes [cm]
    D_size    : (det_cols, det_rows)
    D_z       : detector pixel height [cm]
    SOD, ODD  : source-to-object and object-to-detector distances [cm]
    return_mask : if True, return (filtered_vectors, bool_mask)

    Returns
    -------
    filtered_vectors [, bool_mask]
    """
    D_Z = D_size[1]
    phi = np.arctan2(D_Z * D_z / 2, SOD + ODD)
    xmax_ext = np.max([np.abs(xax[0]), np.abs(xax[-1])])
    ymax_ext = np.max([np.abs(yax[0]), np.abs(yax[-1])])
    maxmax = np.max([xmax_ext, ymax_ext])
    z_margin = (SOD + maxmax) * np.tan(phi)
    zmin_new = zmin - z_margin
    zmax_new = zmax + z_margin
    bool_mask = (vectors[:, 2] >= zmin_new) & (vectors[:, 2] <= zmax_new)
    if return_mask:
        return vectors[bool_mask], bool_mask, z_margin
    return vectors[bool_mask]


#------------------------------------------------------
#-------------- CT sweep execution helpers ------------
#------------------------------------------------------

def generate_sweeps(template_lines, initial_z, z_step, label, out_base,
                    phantom_path, output_dir, results_dir, projections_per_sweep,
                    angle_between, total_projections):
    """Generate .in files for one scan type (blank or phantom).

    Returns (sweep_files, sweep_info) where sweep_info is a list of dicts with
    keys: file, n_proj, source_z, angle_start, z_step.
    """
    print(f"\n--- {label.upper()} SCAN ---")
    print(f"\n{'Sweep':>6} | {'Proj':>5} | {'Start Angle':>12} | "
          f"{'Source Z':>12} | {'Z step':>12} | File")
    print("-" * 80)

    cumulative_angle = 0.0
    cumulative_z = 0.0
    current_translation = z_step
    remaining = total_projections
    sweep = 0
    sweep_files = []
    sweep_info = []

    while remaining > 0:
        n_proj = min(projections_per_sweep, remaining)
        sweep_name = os.path.join(results_dir, f"{out_base}_sweep_{sweep:04d}")
        in_filename = f"{label}_sweep_{sweep:04d}.in"
        in_filepath = os.path.join(output_dir, in_filename)

        lines = list(template_lines)
        source_z = initial_z + cumulative_z
        update_source_z(lines, source_z)
        update_num_projections(lines, n_proj)
        update_angular_rotation_to_first(lines, cumulative_angle)
        update_translation_along_axis(lines, current_translation)
        update_output_name(lines, sweep_name)
        update_angle_between_projections(lines, angle_between)
        update_phantom_file(lines, phantom_path)

        with open(in_filepath, 'w') as f:
            f.writelines(lines)

        print(f"{sweep:6d} | {n_proj:5d} | {cumulative_angle:10.4f}° | "
              f"{source_z:10.4f} cm | {current_translation:+10.4f} cm | "
              f"{in_filename}")

        sweep_files.append(in_filepath)
        sweep_info.append({
            "file": in_filename, "n_proj": n_proj,
            "source_z": source_z, "angle_start": cumulative_angle,
            "z_step": current_translation,
        })

        cumulative_z += (n_proj - 1) * current_translation
        cumulative_angle += (n_proj - 1) * angle_between
        current_translation *= -1
        remaining -= n_proj
        sweep += 1

    return sweep_files, sweep_info


def run_sweeps(sweep_files, mcgpu):
    """Run each .in file via MC-GPU. Returns list of (filepath, returncode)."""
    results = []
    for in_file in sweep_files:
        print(f"  Running: mpirun -n 1 {mcgpu} {in_file}")
        proc = subprocess.run(["mpirun", "-n", "1", mcgpu, in_file])
        results.append((in_file, proc.returncode))
        if proc.returncode != 0:
            print(f"  WARNING: MC-GPU returned {proc.returncode} for {in_file}")
    return results


def select_sweep_subsets(header_files, phantom_nii, template_file, z_step, zmin, zmax):
    """Group blank headers by sweep and find the contiguous Z-region block per sweep.

    Filename pattern expected: {prefix}_sweep_{XXXX}_{YYYY}
      XXXX = sweep index (even → going up, odd → going down)
      YYYY = 1-based projection index within the sweep

    Returns
    -------
    subsets : list of dicts with keys
        sweep_idx, i_start_orig (1-based), n_proj_subset,
        src_pos (3,), direction (3,), z_step_signed
    z_margin : float [cm]
    """
    all_entries = []   # (sweep_idx, proj_idx, source_pos, direction)
    nx, nz = None, None

    for hf in header_files:
        m = re.search(r'_sweep_(\d+)_(\d+)$', os.path.basename(hf))
        if not m:
            continue
        sweep_idx = int(m.group(1))
        proj_idx  = int(m.group(2))   # 1-based

        info = parse_header(hf)
        if nx is None:
            nx, nz = info['nx'], info['nz']
        all_entries.append((sweep_idx, proj_idx, info['source_pos'], info['direction']))

    all_positions = np.array([e[2] for e in all_entries])   # (N, 3)

    params = parse_in_file(template_file)
    D_z    = params['height_z'] / nz
    SOD    = params['sod']
    ODD    = params['sdd'] - SOD
    D_size = (nx, nz)

    _, (_, yax, xax) = read_nii(phantom_nii)
    xax, yax = xax / 10, yax / 10   # mm → cm

    _, bool_mask, z_margin = subsample_slicewise(
        all_positions, xax, yax,
        zmin=zmin, zmax=zmax,
        D_size=D_size, D_z=D_z,
        SOD=SOD, ODD=ODD,
        return_mask=True,
    )

    print(f"  Z margin: {z_margin:.4f} cm  →  effective range "
          f"[{zmin - z_margin:.4f}, {zmax + z_margin:.4f}] cm")

    # Group selected entries by sweep
    sweep_data = {}
    for (sweep_idx, proj_idx, src_pos, direction), keep in zip(all_entries, bool_mask):
        if not keep:
            continue
        sweep_data.setdefault(sweep_idx, []).append((proj_idx, src_pos, direction))

    subsets = []
    total_proj = 0

    for sweep_idx in sorted(sweep_data):
        z_step_signed = z_step if sweep_idx % 2 == 0 else -z_step
        going_up = z_step_signed > 0

        # Sort by source Z in travel direction so index 0 is first to enter region:
        #   going up   → ascending Z  → first = min Z
        #   going down → descending Z → first = max Z
        projs = sorted(sweep_data[sweep_idx],
                       key=lambda x: x[1][2],
                       reverse=not going_up)

        i_start_orig  = projs[0][0]   # 1-based original proj index
        n_proj_subset = len(projs)
        src_pos       = projs[0][1]
        direction     = projs[0][2]

        subsets.append({
            'sweep_idx':     sweep_idx,
            'i_start_orig':  i_start_orig,
            'n_proj_subset': n_proj_subset,
            'src_pos':       src_pos,
            'direction':     direction,
            'z_step_signed': z_step_signed,
        })
        total_proj += n_proj_subset

        direction_label = 'up' if z_step_signed > 0 else 'down'
        print(f"  Sweep {sweep_idx:04d} ({direction_label}): "
              f"proj {i_start_orig:04d}–{i_start_orig + n_proj_subset - 1:04d}  "
              f"({n_proj_subset} projections)")

    print(f"  Total: {len(subsets)} sweeps, {total_proj} projections")
    return subsets, z_margin