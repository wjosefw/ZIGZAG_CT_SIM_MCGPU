from pathlib import Path

import numpy as np

# Map element name → atomic number for lookup during parsing
ELEMENT_Z = {
    "Hydrogen": 1, "Carbon": 6, "Nitrogen": 7, "Oxygen": 8,
    "Magnesium": 12, "Phosphorus": 15, "Sulfur": 16, "Chlorine": 17,
    "Argon": 18, "Calcium": 20, "Sodium": 11, "Potassium": 19,
    "Titanium": 22,
}


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
                {"Z": ELEMENT_Z[name], "weight_fraction": float(w)}
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
