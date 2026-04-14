"""
Generate multiple MC-GPU input files for a zigzag CT scan.

MC-GPU only supports a fixed vertical translation direction per call.
This script splits a full scan into multiple sweeps (alternating up/down),
maintaining rotation continuity across sweeps.
"""

import os
import argparse

from Utils_Sim_Listmode import (
    parse_template,
    get_initial_z,
    get_output_base_name,
    update_source_z,
    update_num_projections,
    update_angular_rotation_to_first,
    update_translation_along_axis,
    update_output_name,
    update_angle_between_projections,
)


def main(template_file, num_sweeps, total_z, z_step, output_dir=None):
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

    main(args.template, args.num_sweeps, args.total_z,
         args.z_step, args.output_dir)
