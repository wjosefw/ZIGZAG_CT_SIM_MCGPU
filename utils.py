from pathlib import Path
import re

import numpy as np
import parallelproj
from array_api_compat import get_namespace, device, to_device

# Map element name → atomic number for lookup during parsing
ELEMENT_Z = {
    "Hydrogen": 1, "Carbon": 6, "Nitrogen": 7, "Oxygen": 8,
    "Magnesium": 12, "Phosphorus": 15, "Sulfur": 16, "Chlorine": 17,
    "Argon": 18, "Calcium": 20, "Sodium": 11, "Potassium": 19,
    "Titanium": 22, "Copper": 29, "Zinc": 30, "Silver": 47, "Tin": 50,
}

ELEMENT_ALIASES = {
    "Phosphor": "Phosphorus",
}


def _element_name_to_z(element_name):
    canonical_name = ELEMENT_ALIASES.get(element_name, element_name)
    if canonical_name not in ELEMENT_Z:
        raise ValueError(
            f"Unknown element '{element_name}' (canonical '{canonical_name}'). "
            "Extend ELEMENT_Z/ELEMENT_ALIASES in utils.py."
        )
    return ELEMENT_Z[canonical_name]


def load_bern_materials(txt_path="Bern_materials.txt"):
    """Load Bern material definitions from text file.

    Returns:
      {
        "materials": [
          {"id", "name", "density_g_cm3", "elements": [{"Z", "weight_fraction"}]}
        ],
        "compositions_by_id": {id: elements},
        "densities_by_id": {id: density_g_cm3},
        "names_by_id": {id: name},
        "hu_material_sections": np.array of HU boundary values (length = n_materials + 1),
      }
    """
    header_re = re.compile(
        r"^([A-Za-z0-9_]+):\s*d=([0-9]*\.?[0-9]+)\s*(mg/cm3|g/cm3)\s*;\s*n=(\d+)\s*;\s*$"
    )
    element_re = re.compile(r"^\+el:\s*name=([^;]+);\s*f=([0-9]*\.?[0-9]+)\s*$")
    section_re = re.compile(r"#\s*Material corresponding to H=\[\s*([+-]?[0-9]*\.?[0-9]+)\s*;\s*([+-]?[0-9]*\.?[0-9]+)\s*\]")

    materials = []
    current = None
    # Collect section boundaries from comment lines; each comment gives [lo, hi]
    # We accumulate unique boundary values in order to build the full boundary array.
    section_boundaries = []
    pending_lo = None

    def finalize_current(material):
        if material is None:
            return
        expected = material["n_expected"]
        found = len(material["elements"])
        if found != expected:
            raise ValueError(
                f"Material '{material['name']}' declares n={expected}, found {found} elements."
            )

        wf_sum = sum(e["weight_fraction"] for e in material["elements"])
        if abs(wf_sum - 1.0) > 1e-3:
            raise ValueError(
                f"Material '{material['name']}' weight fractions sum to {wf_sum:.6f}, expected 1.0."
            )

        material["elements"] = sorted(material["elements"], key=lambda x: x["Z"])
        del material["n_expected"]
        materials.append(material)

    for raw_line in Path(txt_path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            sec_match = section_re.match(line)
            if sec_match:
                lo = float(sec_match.group(1))
                hi = float(sec_match.group(2))
                if not section_boundaries or section_boundaries[-1] != lo:
                    section_boundaries.append(lo)
                pending_lo = hi
            continue

        header_match = header_re.match(line)
        if header_match:
            finalize_current(current)

            name = header_match.group(1)
            density_value = float(header_match.group(2))
            density_unit = header_match.group(3)
            n_expected = int(header_match.group(4))

            if density_unit == "mg/cm3":
                density_g_cm3 = density_value / 1000.0
            elif density_unit == "g/cm3":
                density_g_cm3 = density_value
            else:
                raise ValueError(f"Unsupported density unit '{density_unit}' in line: {line}")

            current = {
                "name": name,
                "density_g_cm3": density_g_cm3,
                "n_expected": n_expected,
                "elements": [],
            }
            continue

        element_match = element_re.match(line)
        if element_match:
            if current is None:
                raise ValueError(f"Found element line before any material header: {line}")
            element_name = element_match.group(1).strip()
            weight_fraction = float(element_match.group(2))
            current["elements"].append(
                {
                    "Z": _element_name_to_z(element_name),
                    "weight_fraction": weight_fraction,
                }
            )
            continue

        raise ValueError(f"Unrecognized Bern material line: {line}")

    finalize_current(current)
    # Append the final upper boundary
    if pending_lo is not None:
        section_boundaries.append(pending_lo)

    for idx, material in enumerate(materials, start=1):
        material["id"] = idx

    compositions_by_id = {m["id"]: m["elements"] for m in materials}
    densities_by_id = {m["id"]: m["density_g_cm3"] for m in materials}
    names_by_id = {m["id"]: m["name"] for m in materials}

    return {
        "materials": materials,
        "compositions_by_id": compositions_by_id,
        "densities_by_id": densities_by_id,
        "names_by_id": names_by_id,
        "hu_material_sections": np.asarray(section_boundaries, dtype=np.float32),
    }


def write_named_material_input(output_dir, materials):
    """Write one .in file per material, using material names for file and .mat names."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for material in materials:
        name = material["name"]
        elements = material["elements"]
        density = material["density_g_cm3"]

        lines = []
        lines.append("1")          # Enter from keyboard
        lines.append(name)         # Material name
        n = len(elements)
        lines.append(str(n))       # Number of elements

        if n == 1:
            # Pure element: no weight fraction prompt
            lines.append(str(elements[0]["Z"]))
        else:
            # Compound: fraction by weight
            lines.append("2")
            for entry in elements:
                lines.append(str(entry["Z"]))
                lines.append(f"{entry['weight_fraction']:.6f}")

        lines.append("2")                  # Don't change I
        lines.append(f"{density:.6f}")     # Density in g/cm3
        lines.append("2")                  # Don't change Fcb/Wcb
        lines.append(f"{name}.mat")        # Output filename

        filepath = out / f"{name}.in"
        filepath.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(materials)} named .in files to {output_dir}/")


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
