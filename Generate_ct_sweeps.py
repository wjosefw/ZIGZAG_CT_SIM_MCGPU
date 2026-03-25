"""
Generate multiple MC-GPU input files for a zigzag CT scan.

MC-GPU only supports a fixed vertical translation direction per call.
This script splits a full scan into multiple sweeps (alternating up/down),
maintaining rotation continuity across sweeps.
"""

import re
import os
import argparse


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


def generate_sweeps(template_file, num_sweeps, total_z, z_step,
                    output_dir=None):
    """Generate .in files for each sweep and a shell script to run them.

    Parameters
    ----------
    template_file : str
        Path to the base MC-GPU .in file.
    num_sweeps : int
        Number of vertical sweeps in a full 360-degree rotation.
    total_z : float
        Total vertical distance per sweep in cm.
    z_step : float
        Vertical translation step between projections in cm.
    output_dir : str or None
        Directory for generated files. Defaults to directory of template_file.
    """
    if output_dir is None:
        output_dir = os.path.dirname(template_file) or '.'
    os.makedirs(output_dir, exist_ok=True)

    template_lines = parse_template(template_file)
    initial_z = get_initial_z(template_lines)
    base_name = get_output_base_name(template_lines)

    # --- Derive quantities ---
    n_steps = round(total_z / z_step)
    projections_per_sweep = n_steps + 1
    angle_per_sweep = 360.0 / num_sweeps
    angle_between = angle_per_sweep / (projections_per_sweep - 1)
    total_projections = projections_per_sweep * num_sweeps

    # --- Print all quantities ---
    print("=" * 60)
    print("  ZIGZAG CT SCAN — PARAMETERS")
    print("=" * 60)
    print(f"  Template file:              {template_file}")
    print(f"  Initial source Z:           {initial_z:.4f} cm")
    print(f"  Output base name:           {base_name}")
    print("-" * 60)
    print(f"  INPUT  num_sweeps:          {num_sweeps}")
    print(f"  INPUT  total_z per sweep:   {total_z:.4f} cm")
    print(f"  INPUT  z_step:              {z_step:.6f} cm")
    print("-" * 60)
    print(f"  DERIVED  steps per sweep:   {n_steps}  "
          f"(round({total_z}/{z_step}) = round({total_z / z_step:.2f}))")
    print(f"  DERIVED  projections/sweep: {projections_per_sweep}  "
          f"(steps + 1 = {n_steps} + 1)")
    print(f"  DERIVED  angle per sweep:   {angle_per_sweep:.6f} deg  "
          f"(360 / {num_sweeps})")
    print(f"  DERIVED  angle between:     {angle_between:.6f} deg  "
          f"({angle_per_sweep:.2f} / ({projections_per_sweep} - 1))")
    print(f"  DERIVED  total projections: {total_projections}  "
          f"({projections_per_sweep} x {num_sweeps})")
    print(f"  DERIVED  total rotation:    "
          f"{total_projections * angle_between:.2f} deg")
    print(f"  Output directory:           {output_dir}")
    print("=" * 60)

    # --- Generate sweep files ---
    cumulative_angle = 0.0
    cumulative_z = 0.0
    current_translation = z_step
    remaining = total_projections
    sweep = 0
    sweep_files = []

    print(f"\n{'Sweep':>6} | {'Proj':>5} | {'Start Angle':>12} | "
          f"{'Source Z':>12} | {'Z step':>12} | File")
    print("-" * 80)

    while remaining > 0:
        n_proj = min(projections_per_sweep, remaining)
        sweep_name = f"{base_name}_sweep_{sweep:04d}"
        in_filename = f"sweep_{sweep:04d}.in"
        in_filepath = os.path.join(output_dir, in_filename)

        # Copy template lines
        lines = list(template_lines)

        # Apply modifications
        source_z = initial_z + cumulative_z
        update_source_z(lines, source_z)
        update_num_projections(lines, n_proj)
        update_angular_rotation_to_first(lines, cumulative_angle)
        update_translation_along_axis(lines, current_translation)
        update_output_name(lines, sweep_name)
        update_angle_between_projections(lines, angle_between)

        with open(in_filepath, 'w') as f:
            f.writelines(lines)

        print(f"{sweep:6d} | {n_proj:5d} | {cumulative_angle:10.4f}° | "
              f"{source_z:10.4f} cm | {current_translation:+10.4f} cm | "
              f"{in_filename}")

        sweep_files.append(in_filepath)

        # Update state for next sweep
        # Z: next sweep starts at the LAST projection's Z (overlap at boundary)
        cumulative_z += (n_proj - 1) * current_translation
        # Angle: next sweep starts at the last projection's angle (overlap)
        cumulative_angle += (n_proj - 1) * angle_between
        current_translation *= -1  # flip direction
        remaining -= n_proj
        sweep += 1

    # Generate shell script
    sh_path = os.path.join(output_dir, "run_ct_scan.sh")
    with open(sh_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated MC-GPU zigzag CT scan runner\n")
        f.write(f"# {sweep} sweeps, {total_projections} total projections\n")
        f.write(f"# Angle between projections: {angle_between:.6f} deg (derived)\n")
        f.write(f"# Z step: {z_step:.6f} cm\n\n")
        for sf in sweep_files:
            f.write(f"./MC-GPU_v1.5b.x {sf}\n")
    os.chmod(sh_path, 0o755)

    print("-" * 80)
    print(f"Generated {sweep} sweep files and {sh_path}")

    return sweep_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MC-GPU input files for a zigzag CT scan.")
    parser.add_argument("template", help="Path to the base MC-GPU .in file")
    parser.add_argument("num_sweeps", type=int,
                        help="Number of vertical sweeps in a full 360-degree rotation")
    parser.add_argument("total_z", type=float,
                        help="Total vertical distance per sweep [cm]")
    parser.add_argument("z_step", type=float,
                        help="Vertical step between projections [cm]")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: same as template)")
    args = parser.parse_args()

    generate_sweeps(args.template, args.num_sweeps, args.total_z,
                    args.z_step, args.output_dir)
