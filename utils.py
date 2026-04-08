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


def parse_xcat_log(log_path):
    """Parse an XCAT phantom log file and return imaging parameters.

    Returns a dict with:
        pixel_width: cm/pixel
        slice_width: cm/pixel
        array_size: number of pixels per side (square array)
        start_slice: first slice number
        end_slice: last slice number
        n_slices: total number of slices
        att_water: linear attenuation coefficient of water (1/cm)
    """
    log_path = Path(log_path)
    params = {}

    for line in log_path.read_text().splitlines():
        stripped = line.strip()

        if stripped.startswith("pixel width"):
            params["pixel_width"] = float(stripped.split("=")[1].split("(")[0].strip())
        elif stripped.startswith("slice width"):
            params["slice_width"] = float(stripped.split("=")[1].split("(")[0].strip())
        elif stripped.startswith("array_size"):
            params["array_size"] = int(stripped.split("=")[1].strip())
        elif stripped.startswith("starting slice number"):
            params["start_slice"] = int(stripped.split("=")[1].strip())
        elif stripped.startswith("ending slice number"):
            params["end_slice"] = int(stripped.split("=")[1].strip())
        elif stripped.startswith("Body (water)") and "(1/cm)" not in stripped:
            # Skip — this is the 1/pixel line
            pass
        elif stripped.startswith("Body (water)"):
            pass

    # Re-parse to get the 1/cm water attenuation (first occurrence of Body (water))
    in_cm_section = False
    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if "Attenuation Coefficients (1/cm)" in stripped:
            in_cm_section = True
        elif "Attenuation Coefficients (1/pixel)" in stripped:
            in_cm_section = False
        elif in_cm_section and stripped.startswith("Body (water)"):
            params["att_water"] = float(stripped.split("=")[1].strip())
            break

    params["n_slices"] = params["end_slice"] - params["start_slice"] + 1

    return params


def load_schneider_materials(txt_path="HUtoMaterialSchneider.txt"):
    """Load Schneider material definitions and return name-keyed compositions.

    Each composition is a list of {"Z": int, "weight_fraction": float} dicts,
    sorted by Z. Only elements with non-zero weight are included.
    """

    lines = Path(txt_path).read_text().splitlines()
    element_names = None
    hu_sections = None
    material_weights = {}

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("sv:Ge/Patient/SchneiderElements"):
            rhs = stripped.split("=", 1)[1].strip()
            tokens = rhs.split()
            expected = int(tokens[0])
            names = [t.strip('"') for t in tokens[1:] if t.strip('"')]
            if len(names) != expected:
                raise ValueError(f"Expected {expected} elements, found {len(names)}")
            element_names = names

        elif stripped.startswith("iv:Ge/Patient/SchneiderHUToMaterialSections"):
            rhs = stripped.split("=", 1)[1].strip()
            tokens = rhs.split()
            expected = int(tokens[0])
            values = [int(v) for v in tokens[1:1 + expected]]
            hu_sections = np.asarray(values, dtype=np.int32)

        elif stripped.startswith("uv:Ge/Patient/SchneiderMaterialsWeight"):
            lhs, rhs = stripped.split("=", 1)
            mid = int(lhs.split("SchneiderMaterialsWeight", 1)[1].strip())
            tokens = rhs.split()
            expected = int(tokens[0])
            material_weights[mid] = np.asarray(tokens[1:1 + expected], dtype=np.float32)

    compositions_by_id = {
        mid: sorted(
            [
                {"Z": _element_name_to_z(name), "weight_fraction": float(w)}
                for name, w in zip(element_names, material_weights[mid])
                if w > 0
            ],
            key=lambda x: x["Z"],
        )
        for mid in sorted(material_weights)
    }

    return {
        "element_names": element_names,
        "hu_material_sections": hu_sections,
        "compositions_by_id": compositions_by_id,
    }


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
      }
    """
    header_re = re.compile(
        r"^([A-Za-z0-9_]+):\s*d=([0-9]*\.?[0-9]+)\s*(mg/cm3|g/cm3)\s*;\s*n=(\d+)\s*;\s*$"
    )
    element_re = re.compile(r"^\+el:\s*name=([^;]+);\s*f=([0-9]*\.?[0-9]+)\s*$")

    materials = []
    current = None

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
        if not line or line.startswith("#"):
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
    }


def load_schneider_density_corrections(txt_path):
    """Load per-HU density corrections from HUtoMaterialSchneider.txt.

    The file stores 3996 correction factors for HU values from -1000 to 2995.
    """
    for line in Path(txt_path).read_text().splitlines():
        if line.startswith("dv:Ge/Patient/DensityCorrection ="):
            values = line.split("=", 1)[1].split()
            expected_count = int(values[0])
            numeric_values = np.asarray(values[1:1 + expected_count], dtype=np.float32)
            if numeric_values.size != expected_count:
                raise ValueError(
                    f"Expected {expected_count} Schneider density corrections, "
                    f"found {numeric_values.size}."
                )
            return numeric_values

    raise ValueError("DensityCorrection line not found in HUtoMaterialSchneider.txt")


def hu_to_density_schneider(hu, txt_path="HUtoMaterialSchneider.txt"):
    """Convert HU to density using the Schneider definition from HUtoMaterialSchneider.txt.

    Base formula:
        density = offset + factor * (factor_offset + HU)

    Then apply the per-HU DensityCorrection lookup loaded from the text file.
    """
    hu = np.asarray(hu, dtype=np.float32)
    density = np.zeros_like(hu, dtype=np.float32)

    sections = np.array([-1000, -98, 15, 23, 101, 2001, 2995, 2996], dtype=np.float32)
    offsets = np.array([0.00121, 1.018, 1.03, 1.003, 1.017, 2.201, 4.54], dtype=np.float32)
    factors = np.array([0.001029700665188, 0.000893, 0.0, 0.001169, 0.000592, 0.0005, 0.0], dtype=np.float32)
    factor_offsets = np.array([1000.0, 0.0, 1000.0, 0.0, 0.0, -2000.0, 0.0], dtype=np.float32)

    for i in range(len(offsets)):
        lower = sections[i]
        upper = sections[i + 1]
        if i < len(offsets) - 1:
            mask = (hu >= lower) & (hu < upper)
        else:
            mask = (hu >= lower) & (hu <= upper)
        density[mask] = offsets[i] + factors[i] * (factor_offsets[i] + hu[mask])

    density_corrections = load_schneider_density_corrections(txt_path)
    hu_index = np.clip(np.rint(hu).astype(np.int32), -1000, 2995) + 1000
    density *= density_corrections[hu_index]

    return density


def write_material_input(output_dir, material_data):
    """Write one .in file per material that answers the interactive prompts of material.x.

    For compounds (>1 element):
      1              # Enter composition from keyboard
      MATERIAL<id>   # Material name
      <n_elements>   # Number of elements
      2              # Fraction by weight
      <Z>            # Atomic number of element
      <w>            # Weight fraction of element
      ...            # Repeat for each element
      2              # Do not change calculated I
      <density>      # Mass density (g/cm3)
      2              # Do not change Fcb/Wcb
      <filename>     # Output .mat filename

    For pure elements (1 element):
      1              # Enter composition from keyboard
      MATERIAL<id>   # Material name
      1              # Number of elements
      <Z>            # Atomic number of the element
      2              # Do not change calculated I
      <density>      # Mass density (g/cm3)
      2              # Do not change Fcb/Wcb
      <filename>     # Output .mat filename
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for mid, elements in material_data.items():
        lines = []
        lines.append("1")                          # Enter from keyboard
        lines.append(f"MATERIAL{mid}")             # Material name
        n = len(elements)
        lines.append(str(n))                       # Number of elements

        if n == 1:
            # Pure element: no weight fraction prompt
            lines.append(str(elements[0]["Z"]))
        else:
            # Compound: fraction by weight
            lines.append("2")
            for entry in elements:
                lines.append(str(entry["Z"]))
                lines.append(f"{entry['weight_fraction']:.6f}")

        lines.append("2")                          # Don't change I
        lines.append("1")                          # Density placeholder
        lines.append("2")                          # Don't change Fcb/Wcb
        lines.append(f"MATERIAL{mid}.mat")         # Output filename

        filepath = out / f"MATERIAL{mid}.in"
        filepath.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(material_data)} .in files to {output_dir}/")


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


def mlem(xstart, xend, measurements, img_dim, img_origin, voxel_size, num_iter=50):
    """MLEM reconstruction from listmode ray data.

    Backend- and device-agnostic: infers the array namespace and device
    from the input arrays, so it works with numpy (CPU), cupy (GPU), or torch.
    """
    xp = get_namespace(measurements)
    dev = device(xstart)

    sensitivity = parallelproj.joseph3d_back(
        xstart, xend, img_dim, img_origin, voxel_size,
        xp.ones(measurements.shape, dtype=xp.float32)
    )
    sensitivity[sensitivity == 0] = 1e-10
    x = to_device(xp.ones(img_dim, dtype=xp.float32), dev)

    for i in range(num_iter):
        proj = parallelproj.joseph3d_fwd(xstart, xend, x, img_origin, voxel_size)
        corr = measurements / (proj + 1e-6)
        backw = parallelproj.joseph3d_back(xstart, xend, img_dim, img_origin, voxel_size, corr)
        x = x * backw / (sensitivity + 1e-6)
        print(f"MLEM iteration {(i + 1):03} / {num_iter:03}  max={x.max():.4f}", end="\r")

    print(f"\nDone. Recon max: {x.max():.4f}")
    return x
