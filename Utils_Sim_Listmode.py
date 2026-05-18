#------------------------------------------------------
#----------------- List mode helpers ------------------
#------------------------------------------------------


import os
import re
import subprocess
import numpy as np
import Sim_config


# ---------------------------------------------------------------------------
# MC-GPU .in file writer
# ---------------------------------------------------------------------------

_VALID_OVERRIDES = frozenset(k.lower() for k in vars(Sim_config) if k.isupper())


def write_in_file(path, overrides=None):
    """Write a valid MC-GPU .in file to *path*.

    All values come from Sim_config. Any key in *overrides* replaces the
    corresponding Sim_config value for this file only.
    """
    if overrides is None:
        overrides = {}

    unknown = set(overrides) - _VALID_OVERRIDES
    if unknown:
        raise ValueError(f"Unknown override key(s): {sorted(unknown)}")

    def g(key):
        return overrides.get(key, getattr(Sim_config, key.upper()))

    with open(path, 'w') as f:
        f.write("\n")
        f.write("# \n")
        f.write("# >>>> INPUT FILE FOR MC-GPU v1.5 VICTRE-DBT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        f.write("#\n")
        f.write("\n")

        f.write("#[SECTION SIMULATION CONFIG v.2009-05-12]\n")
        f.write(f"{g('n_histories')}                        # TOTAL NUMBER OF HISTORIES, OR SIMULATION TIME IN SECONDS IF VALUE < 100000\n")
        f.write(f"{g('random_seed')}            # RANDOM SEED (ranecu PRNG)\n")
        f.write(f"{g('gpu_number')}                               # GPU NUMBER TO USE WHEN MPI IS NOT USED, OR TO BE AVOIDED IN MPI RUNS\n")
        f.write(f"{g('gpu_threads_per_block')}                             # GPU THREADS PER CUDA BLOCK (multiple of 32)\n")
        f.write(f"{g('histories_per_thread')}                          # SIMULATED HISTORIES PER GPU THREAD\n")
        f.write(" \n")

        f.write("#[SECTION SOURCE v.2016-12-02]\n")
        f.write(f"{g('spectrum_file')} # X-RAY ENERGY SPECTRUM FILE\n")
        f.write(f"{g('source_x')}  {g('source_y')}   {g('source_z')}            # SOURCE POSITION: X (chest-to-nipple), Y (right-to-left), Z (caudal-to-cranial) [cm]\n")
        f.write(f" {g('source_dir_u')}    {g('source_dir_v')}    {g('source_dir_w')}             # SOURCE DIRECTION COSINES: U V W\n")
        f.write(f"{g('beam_aperture_azimuthal')}    {g('beam_aperture_polar')}    # TOTAL AZIMUTHAL (WIDTH, X) AND POLAR (HEIGHT, Z) APERTURES OF THE FAN BEAM [degrees] (input negative to automatically cover the whole detector)\n")
        f.write(f"{g('euler_alpha')}   {g('euler_beta')}  {g('euler_gamma')}             # EULER ANGLES (RzRyRz) TO ROTATE RECTANGULAR BEAM FROM DEFAULT POSITION AT Y=0, NORMAL=(0,-1,0)\n")
        f.write(f" {g('focal_spot_fwhm')}                      # SOURCE GAUSSIAN FOCAL SPOT FWHM [cm]\n")
        f.write(f" {g('angular_blur')}                              # ANGULAR BLUR DUE TO MOVEMENT ([exposure_time]*[angular_speed]) [degrees]\n")
        f.write(f"{g('collimate_beam')}                             # COLLIMATE BEAM TOWARDS POSITIVE AZIMUTHAL (X) ANGLES ONLY? (ie, cone-beam center aligned with chest wall in mammography) [YES/NO]\n")
        f.write(" \n")

        f.write("#[SECTION IMAGE DETECTOR v.2017-06-20]\n")
        f.write(f"{g('output_name')}    # OUTPUT IMAGE FILE NAME\n")
        f.write(f"{g('detector_nx')}   {g('detector_nz')}                  # NUMBER OF PIXELS IN THE IMAGE: Nx Nz\n")
        f.write(f"{g('detector_width_x')}    {g('detector_height_z')}                 # IMAGE SIZE (width, height): Dx Dz [cm]\n")
        f.write(f"{g('sdd')}                           # SOURCE-TO-DETECTOR DISTANCE (detector set in front of the source, perpendicular to the initial direction)\n")
        f.write(f" {g('image_offset_x')}    {g('image_offset_z')}                     # IMAGE OFFSET ON DETECTOR PLANE IN WIDTH AND HEIGHT DIRECTIONS (BY DEFAULT BEAM CENTERED AT IMAGE CENTER) [cm]\n")
        f.write(f" {g('detector_thickness')}                         # DETECTOR THICKNESS [cm]\n")
        f.write(f" {g('detector_mfp')}  # ==> MFP(Se,19.0keV)   # DETECTOR MATERIAL MEAN FREE PATH AT AVERAGE ENERGY [cm]\n")
        f.write(f" {g('detector_kedge_energy')} {g('detector_kfluor_energy')} {g('detector_kfluor_yield')} {g('detector_kfluor_mfp')}  # DETECTOR K-EDGE ENERGY [eV], K-FLUORESCENCE ENERGY [eV], K-FLUORESCENCE YIELD, MFP AT FLUORESCENCE ENERGY [cm]\n")
        f.write(f" {g('detector_gain')}    {g('swank_factor')}                   # EFECTIVE DETECTOR GAIN, W_+- [eV/ehp], AND SWANK FACTOR (input 0 to report ideal energy fluence)\n")
        f.write(f" {g('electronic_noise')}                         # ADDITIVE ELECTRONIC NOISE LEVEL (electrons/pixel)\n")
        f.write(f" {g('cover_thickness')}  {g('cover_mfp')}          # ==> MFP(polystyrene,19keV)       # PROTECTIVE COVER THICKNESS (detector+grid) [cm], MEAN FREE PATH AT AVERAGE ENERGY [cm]\n")
        f.write(f" {g('grid_ratio')}   {g('grid_frequency')}   {g('grid_strip_thickness')}            # ANTISCATTER GRID RATIO, FREQUENCY, STRIP THICKNESS [X:1, lp/cm, cm] (enter 0 to disable the grid)\n")
        f.write(f" {g('grid_strips_mfp')}   {g('grid_interspace_mfp')}   # ==> MFP(lead&polystyrene,19keV)  # ANTISCATTER STRIPS AND INTERSPACE MEAN FREE PATHS AT AVERAGE ENERGY [cm]\n")
        f.write(f" {g('grid_orientation')}                              # ORIENTATION 1D FOCUSED ANTISCATTER GRID LINES: 0==STRIPS PERPENDICULAR LATERAL DIRECTION (mammo style); 1==STRIPS PARALLEL LATERAL DIRECTION (DBT style)\n")
        f.write("\n")

        f.write("#[SECTION TOMOGRAPHIC TRAJECTORY v.2016-12-02]\n")
        f.write(f"{g('n_projections')}     # ==> 1 for mammo only; ==> 25 for mammo + DBT    # NUMBER OF PROJECTIONS (1 disables the tomographic mode)\n")
        f.write(f"{g('sod')}                            # SOURCE-TO-ROTATION AXIS DISTANCE\n")
        f.write(f"{g('angle_between_projections')}          # ANGLE BETWEEN PROJECTIONS (360/num_projections for full CT) [degrees]\n")
        f.write(f"{g('rotation_to_first_projection')}                           # ANGULAR ROTATION TO FIRST PROJECTION (USEFUL FOR DBT, INPUT SOURCE DIRECTION CONSIDERED AS 0 DEGREES) [degrees]\n")
        f.write(f"{g('rot_axis_x')}  {g('rot_axis_y')}  {g('rot_axis_z')}                  # AXIS OF ROTATION (Vx,Vy,Vz)\n")
        f.write(f"{g('translation_along_axis')}                           # TRANSLATION ALONG ROTATION AXIS BETWEEN PROJECTIONS (HELICAL SCAN) [cm]\n")
        f.write(f"{g('keep_detector_fixed')}                             # KEEP DETECTOR FIXED AT 0 DEGREES FOR DBT? [YES/NO]\n")
        f.write(f"{g('simulate_both')}                            # SIMULATE BOTH 0 deg PROJECTION AND TOMOGRAPHIC SCAN (WITHOUT GRID) WITH 2/3 TOTAL NUM HIST IN 1st PROJ (eg, DBT+mammo)? [YES/NO]\n")
        f.write("\n")

        f.write("#[SECTION DOSE DEPOSITION v.2012-12-12]\n")
        f.write(f"{g('tally_material_dose')}                              # TALLY MATERIAL DOSE? [YES/NO] (disabled for crash isolation)\n")
        f.write(f"{g('tally_voxel_dose')}                              # TALLY 3D VOXEL DOSE? [YES/NO] (dose measured separately for each voxel)\n")
        f.write(f"{g('dose_output_file')}                 # OUTPUT VOXEL DOSE FILE NAME\n")
        f.write(f"  {g('dose_roi_x_min')} {g('dose_roi_x_max')}                         # VOXEL DOSE ROI: X-index min max (must be within phantom Nx=512)\n")
        f.write(f"  {g('dose_roi_y_min')} {g('dose_roi_y_max')}                         # VOXEL DOSE ROI: Y-index min max (must be within phantom Ny=512)\n")
        f.write(f"{g('dose_roi_z_min')}  {g('dose_roi_z_max')}                        # VOXEL DOSE ROI: Z-index min max\n")
        f.write(" \n")

        f.write("#[SECTION VOXELIZED GEOMETRY FILE v.2017-07-26]\n")
        f.write(f"{g('phantom_file')}\n")
        f.write(f" {g('phantom_offset_x')}    {g('phantom_offset_y')}    {g('phantom_offset_z')}              # OFFSET OF THE VOXEL GEOMETRY (DEFAULT ORIGIN AT LOWER BACK CORNER) [cm]\n")
        f.write(f" {g('phantom_nx')}    {g('phantom_ny')}    {g('phantom_nz')}              # NUMBER OF VOXELS: INPUT A 0 TO READ ASCII FORMAT WITH HEADER SECTION, RAW VOXELS WILL BE READ OTHERWISE\n")
        f.write(f"{g('voxel_size_x')} {g('voxel_size_y')} {g('voxel_size_z')}   # VOXEL SIZES [cm]\n")
        f.write(f" {g('tree_nx')} {g('tree_ny')} {g('tree_nz')}                        # SIZE OF LOW RESOLUTION VOXELS THAT WILL BE DESCRIBED BY A BINARY TREE, GIVEN AS POWERS OF TWO (eg, 2 2 3 = 2^2x2^2x2^3 = 128 input voxels per low res voxel; 0 0 0 disables tree)\n")
        f.write(" \n")

        f.write("#[SECTION MATERIAL FILE LIST v.2009-11-30]\n")
        for i, (mat_file, density, voxel_id) in enumerate(g('materials'), start=1):
            f.write(f"{mat_file}    density={density}     voxelId={voxel_id}       #   {i:2d}th MATERIAL FILE\n")


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


def compute_euler_angles(dir_x, dir_y, dir_z):
    """RzRyRz Euler angles rotating MC-GPU default direction (0,1,0) to (dir_x, dir_y, dir_z).
    Valid when dir_z ≈ 0. Returns (alpha, beta, gamma) in degrees.
    """
    alpha = np.degrees(np.arctan2(dir_x, -dir_y))
    beta  = 0.0
    gamma = 180.0
    return alpha, beta, gamma


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

def generate_sweeps(z_step, label, out_base,
                    phantom_path, output_dir, results_dir, projections_per_sweep,
                    angle_between, total_projections):
    """Generate .in files for one scan type (blank or phantom).

    Returns (sweep_files, sweep_info) where sweep_info is a list of dicts with
    keys: file, n_proj, source_z, angle_start, z_step.
    """
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

        source_z = Sim_config.SOURCE_Z + cumulative_z
        write_in_file(in_filepath, {
            "source_z":                     source_z,
            "n_projections":                n_proj,
            "rotation_to_first_projection": cumulative_angle,
            "translation_along_axis":       current_translation,
            "output_name":                  sweep_name,
            "angle_between_projections":    angle_between,
            "phantom_file":                 phantom_path,
        })

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
        proc = subprocess.run(["mpirun", "-n", "1", mcgpu, in_file])
        results.append((in_file, proc.returncode))
        if proc.returncode != 0:
            print(f"  WARNING: MC-GPU returned {proc.returncode} for {in_file}")
    return results


def select_sweep_subsets(header_files, z_step, zmin, zmax):
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

    D_z    = Sim_config.DETECTOR_HEIGHT_Z / nz
    SOD    = Sim_config.SOD
    ODD    = Sim_config.SDD - SOD
    D_size = (nx, nz)

    xax = np.array([Sim_config.PHANTOM_OFFSET_X, Sim_config.PHANTOM_OFFSET_X + Sim_config.PHANTOM_NX * Sim_config.VOXEL_SIZE_X])
    yax = np.array([Sim_config.PHANTOM_OFFSET_Y, Sim_config.PHANTOM_OFFSET_Y + Sim_config.PHANTOM_NY * Sim_config.VOXEL_SIZE_Y])

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