"""Generate .in and .mat files for PENELOPE material.x.

Material data is imported directly from bern_materials_db or gate_materials_db
— no text-file parsing at runtime.

Usage example
-------------
python Generate_PENELOPE_materials_Bern.py --source bern \\
    --output-dir   pendbase/ \\
    --pendbase-dir pendbase/

python Generate_PENELOPE_materials_Bern.py --source gate \\
    --output-dir   pendbase/ \\
    --pendbase-dir pendbase/
"""

import argparse

from bern_materials_db import MATERIALS as BERN_MATERIALS
from gate_materials_db import MATERIALS as GATE_MATERIALS
from Utils_Materials import run_material_creation, write_named_material_input

SOURCES = {
    "bern": BERN_MATERIALS,
    "gate": GATE_MATERIALS,
}


def main(source, output_dir, pendbase_dir):
    materials = SOURCES[source]
    print(f"Loaded {len(materials)} materials from {source}_materials_db")

    sample = materials[0]
    print(
        f"Example: {sample['name']} "
        f"(density={sample['density_g_cm3']} g/cm3, "
        f"elements={len(sample['elements'])})"
    )

    write_named_material_input(output_dir=output_dir, materials=materials)

    run_material_creation(
        input_dir=output_dir,
        output_dir=output_dir,
        pendbase_dir=pendbase_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["bern", "gate"],
        default="bern",
        help="Material database to use (default: bern)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write .in and .mat files",
    )
    parser.add_argument(
        "--pendbase-dir",
        required=True,
        help="Directory containing material.x and PENELOPE data files",
    )
    args = parser.parse_args()
    main(
        source=args.source,
        output_dir=args.output_dir,
        pendbase_dir=args.pendbase_dir,
    )
