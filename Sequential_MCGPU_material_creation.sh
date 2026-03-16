#!/bin/bash
# Run MC-GPU_create_material_data.x for each .mat file

OUTDIR="${1:-.}"

if [ ! -d "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
    echo "Created output directory: $OUTDIR"
fi

for matfile in pendbase/MATERIAL*.mat; do
    basename="${matfile##*/}"
    name="${basename%.mat}"
    outfile="${OUTDIR}/${name}_5_150keV.mcgpu"
    echo "Processing $matfile -> $outfile ..."
    cat <<EOF | ./MC-GPU_create_material_data.x
5000,150000
1495
${matfile}
${outfile}
EOF
done

echo "All MC-GPU material data files created."
