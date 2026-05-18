"""
Generate multiple MC-GPU input files for a zigzag CT scan and run them.

MC-GPU only supports a fixed vertical translation direction per call.
This script splits a full scan into multiple sweeps (alternating up/down),
maintaining rotation continuity across sweeps.

Modes
-----
full   (default): generate blank + phantom .in files, run all sweeps, write log.
subset:           run blank sweeps, parse their headers to find which projections
                  per sweep fall inside a Z range, copy matching blank files, then
                  re-simulate each qualifying sweep as one contiguous MC-GPU call
                  (one initialisation per sweep, not per projection).
"""

import os
import glob
import shutil
import argparse
import subprocess

import Sim_config
from mcgpu_writer import write_in_file
from Utils_Sim_Listmode import (
    compute_euler_angles,
    generate_sweeps,
    run_sweeps,
    select_sweep_subsets,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(num_sweeps, total_z, z_step,
         phantom_path, phantom_output, blank_path, blank_output,
         mcgpu, results_dir, mode='full',
         zmin=-55.0, zmax=55.0,
         output_dir=None):

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    n_steps = round(total_z / z_step)
    projections_per_sweep = n_steps + 1
    angle_per_sweep = 360.0 / num_sweeps
    angle_between = angle_per_sweep / (projections_per_sweep - 1)
    total_projections = projections_per_sweep * num_sweeps

    param_summary = "\n".join([
        "=" * 60,
        "  ZIGZAG CT SCAN — PARAMETERS",
        "=" * 60,
        f"  Config:                     sim_config.py",
        f"  Mode:                       {mode}",
        f"  Initial source Z:           {Sim_config.SOURCE_Z:.4f} cm",
        f"  Phantom geometry:           {phantom_path}",
        f"  Phantom output base:        {phantom_output}",
        f"  Blank geometry:             {blank_path}",
        f"  Blank output base:          {blank_output}",
        f"  MC-GPU executable:          {mcgpu}",
        f"  Results directory:          {results_dir}",
        "-" * 60,
        f"  INPUT  num_sweeps:          {num_sweeps}",
        f"  INPUT  total_z per sweep:   {total_z:.4f} cm",
        f"  INPUT  z_step:              {z_step:.6f} cm",
        "-" * 60,
        f"  DERIVED  steps per sweep:   {n_steps}  "
            f"(round({total_z}/{z_step}) = round({total_z / z_step:.2f}))",
        f"  DERIVED  projections/sweep: {projections_per_sweep}  "
            f"(steps + 1 = {n_steps} + 1)",
        f"  DERIVED  angle per sweep:   {angle_per_sweep:.6f} deg  "
            f"(360 / {num_sweeps})",
        f"  DERIVED  angle between:     {angle_between:.6f} deg  "
            f"({angle_per_sweep:.2f} / ({projections_per_sweep} - 1))",
        f"  DERIVED  total projections: {total_projections}  "
            f"({projections_per_sweep} x {num_sweeps})",
        f"  DERIVED  total rotation:    {total_projections * angle_between:.2f} deg",
        f"  Output directory:           {output_dir}",
        "=" * 60,
    ])

    # Generate blank sweep files (both modes need them)
    blank_files, blank_sweep_info = generate_sweeps(
        z_step,
        "blank", blank_output, blank_path, output_dir, results_dir,
        projections_per_sweep, angle_between, total_projections,
    )

    # -----------------------------------------------------------------------
    # FULL MODE
    # -----------------------------------------------------------------------
    if mode == 'full':
        phantom_files, phantom_sweep_info = generate_sweeps(
            z_step,
            "phantom", phantom_output, phantom_path, output_dir, results_dir,
            projections_per_sweep, angle_between, total_projections,
        )

        # Run sweeps
        blank_results = run_sweeps(blank_files, mcgpu)
        phantom_results = run_sweeps(phantom_files, mcgpu)

        log_path = os.path.join(output_dir, "generate_ct_sweeps.log")
        with open(log_path, 'w') as f:
            f.write(param_summary + "\n\n")
            f.write("BLANK SWEEPS:\n")
            for info, (fp, rc) in zip(blank_sweep_info, blank_results):
                f.write(f"  {info['file']}  n_proj={info['n_proj']}  "
                        f"source_z={info['source_z']:.4f}  "
                        f"angle_start={info['angle_start']:.4f}  "
                        f"z_step={info['z_step']:+.6f}  rc={rc}\n")
            f.write("PHANTOM SWEEPS:\n")
            for info, (fp, rc) in zip(phantom_sweep_info, phantom_results):
                f.write(f"  {info['file']}  n_proj={info['n_proj']}  "
                        f"source_z={info['source_z']:.4f}  "
                        f"angle_start={info['angle_start']:.4f}  "
                        f"z_step={info['z_step']:+.6f}  rc={rc}\n")
        print(f"\nLog written to {log_path}")

    # -----------------------------------------------------------------------
    # SUBSET MODE
    # -----------------------------------------------------------------------
    elif mode == 'subset':
        # Run blank sweeps
        run_sweeps(blank_files, mcgpu)

        # Glob blank headers produced by MC-GPU
        pattern = os.path.join(results_dir, f"{blank_output}_sweep_*")
        all_files = sorted(glob.glob(pattern))
        header_files = [f for f in all_files if not f.endswith('.raw')]
        print(f"\nFound {len(header_files)} blank headers in {results_dir}")

        # Group by sweep, find contiguous block per sweep inside Z region
        subsets, z_margin = select_sweep_subsets(
            header_files, z_step, zmin, zmax,
        )

        blank_subset_prefix   = f"{blank_output}_subset"
        phantom_subset_prefix = f"{phantom_output}_subset"

        # Per-sweep: copy blank files + simulate phantom as one contiguous run
        for sub in subsets:
            sweep_idx     = sub['sweep_idx']
            i_start_orig  = sub['i_start_orig']
            n_proj_subset = sub['n_proj_subset']
            src_pos       = sub['src_pos']
            direction     = sub['direction']
            z_step_signed = sub['z_step_signed']

            # Copy blank files, renaming to 1-based subset indices
            print(f"\n  Sweep {sweep_idx:04d}: copying {n_proj_subset} blank files")
            for j in range(n_proj_subset):
                orig_idx = i_start_orig + j
                src = os.path.join(results_dir,
                                   f"{blank_output}_sweep_{sweep_idx:04d}_{orig_idx:04d}")
                dst = os.path.join(results_dir,
                                   f"{blank_subset_prefix}_sweep_{sweep_idx:04d}_{j+1:04d}")
                shutil.copy2(src,          dst)
                shutil.copy2(src + ".raw", dst + ".raw")

            # Build phantom .in for this sweep subset and run
            src_x, src_y, src_z = src_pos
            dir_x, dir_y, dir_z = direction
            alpha, beta, gamma  = compute_euler_angles(dir_x, dir_y, dir_z)

            out_name = os.path.join(results_dir,
                                    f"{phantom_subset_prefix}_sweep_{sweep_idx:04d}")
            tmp_in   = os.path.join(output_dir, f"subset_tmp_{sweep_idx:04d}.in")

            write_in_file(tmp_in, {
                "n_projections":                n_proj_subset,
                "source_x":                     src_x,
                "source_y":                     src_y,
                "source_z":                     src_z,
                "source_dir_u":                 dir_x,
                "source_dir_v":                 dir_y,
                "source_dir_w":                 dir_z,
                "euler_alpha":                  alpha,
                "euler_beta":                   beta,
                "euler_gamma":                  gamma,
                "rotation_to_first_projection": 0.0,
                "translation_along_axis":       z_step_signed,
                "angle_between_projections":    angle_between,
                "output_name":                  out_name,
                "phantom_file":                 phantom_path,
            })

            direction_label = 'up' if z_step_signed > 0 else 'down'
            print(f"  Sweep {sweep_idx:04d} ({direction_label}): "
                  f"{n_proj_subset} proj  "
                  f"src=({src_x:.3f},{src_y:.3f},{src_z:.3f})  "
                  f"z_step={z_step_signed:+.4f} cm")

            try:
                subprocess.run(["mpirun", "-n", "1", mcgpu, tmp_in], check=True)
            finally:
                os.remove(tmp_in)

        # Write log
        total_proj = sum(s['n_proj_subset'] for s in subsets)
        log_path = os.path.join(output_dir, "generate_ct_sweeps_subset.log")
        with open(log_path, 'w') as f:
            f.write(param_summary + "\n\n")
            f.write(f"zmin={zmin}  zmax={zmax}  z_margin={z_margin:.4f} cm\n")
            f.write(f"Total blank headers: {len(header_files)}\n")
            f.write(f"Sweeps simulated: {len(subsets)}  "
                    f"Total projections: {total_proj}\n")
            f.write(f"Blank subset prefix:   {blank_subset_prefix}\n")
            f.write(f"Phantom subset prefix:  {phantom_subset_prefix}\n\n")
            f.write("SWEEP SUBSETS:\n")
            for sub in subsets:
                pos, d = sub['src_pos'], sub['direction']
                f.write(f"  sweep {sub['sweep_idx']:04d}  "
                        f"i_start={sub['i_start_orig']:04d}  "
                        f"n_proj={sub['n_proj_subset']}  "
                        f"z_step={sub['z_step_signed']:+.6f}  "
                        f"src=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f})  "
                        f"dir=({d[0]:.4f},{d[1]:.4f},{d[2]:.4f})\n")
        print(f"\nLog written to {log_path}")

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'full' or 'subset'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MC-GPU input files and run zigzag CT sweeps.")
    parser.add_argument("--num-sweeps",  type=int, required=True,
                        help="Number of vertical sweeps in a full 360-degree rotation")
    parser.add_argument("--total-z",     type=float, required=True,
                        help="Total vertical distance per sweep [cm]")
    parser.add_argument("--z-step",      type=float, required=True,
                        help="Vertical step between projections [cm]")
    parser.add_argument("--phantom-path",   required=True,
                        help="Voxelized geometry file for the phantom scan")
    parser.add_argument("--phantom-output", required=True,
                        help="Output image base name for the phantom scan")
    parser.add_argument("--blank-path",     required=True,
                        help="Voxelized geometry file for the blank scan")
    parser.add_argument("--blank-output",   required=True,
                        help="Output image base name for the blank scan")
    parser.add_argument("--mcgpu",          required=True,
                        help="Path to MC-GPU executable (e.g. ./MC-GPU_v1.5b.x)")
    parser.add_argument("--results-dir",    required=True,
                        help="Directory where MC-GPU writes its output files")
    parser.add_argument("--mode",           choices=["full", "subset"], default="full",
                        help="Execution mode (default: full)")
    parser.add_argument("--zmin",           type=float, default=-55.0,
                        help="Min source Z for subset selection [cm] (default: -55)")
    parser.add_argument("--zmax",           type=float, default=55.0,
                        help="Max source Z for subset selection [cm] (default: +55)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory for generated .in files (default: same as template)")
    args = parser.parse_args()

    main(
        num_sweeps     = args.num_sweeps,
        total_z        = args.total_z,
        z_step         = args.z_step,
        phantom_path   = args.phantom_path,
        phantom_output = args.phantom_output,
        blank_path     = args.blank_path,
        blank_output   = args.blank_output,
        mcgpu          = args.mcgpu,
        results_dir    = args.results_dir,
        mode           = args.mode,
        zmin           = args.zmin,
        zmax           = args.zmax,
        output_dir     = args.output_dir,
    )
