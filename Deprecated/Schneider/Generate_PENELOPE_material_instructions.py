"""Generate .in input files for the material definition script.

Loads Schneider material compositions, builds {element: Z, weight_fraction}
dictionaries, and writes .in files that answer the interactive prompts automatically.
"""

import argparse

from Deprecated.Schneider.utils import load_schneider_materials, write_material_input

def main(output_dir):
    schneider = load_schneider_materials()
    print(f"Loaded {len(schneider['compositions_by_id'])} materials")

    # compositions_by_id contains lists of {Z, weight_fraction} dicts
    material_data = schneider["compositions_by_id"]

    # Show one example
    print(f"Material 10 ({len(material_data[10])} elements):")
    for entry in material_data[10]:
        print(f"  Z={entry['Z']:>2d}  w={entry['weight_fraction']:.5f}")

    write_material_input(output_dir, material_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory to write .in files to")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
