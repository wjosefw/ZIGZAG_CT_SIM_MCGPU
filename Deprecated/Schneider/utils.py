from pathlib import Path

import numpy as np

# Map element name -> atomic number for lookup during parsing
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
            "Extend ELEMENT_Z/ELEMENT_ALIASES in Deprecated/Schneider/utils.py."
        )
    return ELEMENT_Z[canonical_name]


def parse_xcat_log(log_path):
    """Parse an XCAT phantom log file and return imaging parameters."""
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
            # Skip - this is the 1/pixel line
            pass
        elif stripped.startswith("Body (water)"):
            pass

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
    """Load Schneider material definitions and return name-keyed compositions."""
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


def load_schneider_density_corrections(txt_path):
    """Load per-HU density corrections from HUtoMaterialSchneider.txt."""
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
    """Convert HU to density using the Schneider definition from HUtoMaterialSchneider.txt."""
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
    """Write one .in file per material for PENELOPE material.x."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    for mid, elements in material_data.items():
        lines = []
        lines.append("1")
        lines.append(f"MATERIAL{mid}")
        n = len(elements)
        lines.append(str(n))

        if n == 1:
            lines.append(str(elements[0]["Z"]))
        else:
            lines.append("2")
            for entry in elements:
                lines.append(str(entry["Z"]))
                lines.append(f"{entry['weight_fraction']:.6f}")

        lines.append("2")
        lines.append("1")
        lines.append("2")
        lines.append(f"MATERIAL{mid}.mat")

        filepath = out / f"MATERIAL{mid}.in"
        filepath.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(material_data)} .in files to {output_dir}/")
