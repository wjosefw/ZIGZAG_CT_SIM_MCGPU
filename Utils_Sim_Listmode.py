#------------------------------------------------------
#----------------- List mode helpers ------------------
#------------------------------------------------------


import numpy as np
import re


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

    # Line 59: AXIS OF ROTATION
    tokens = first_token(lines[58])
    rot_axis = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])])
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    return {
        "sdd": sdd,
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