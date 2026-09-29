#!/usr/bin/env bash

in_dir=../../results/04_single_cell_access_atac/06_add_barcodes
out_dir=../../results/04_single_cell_access_atac/08_bam_to_fragment

# Create output directory if it does not exist
if [ ! -d ${out_dir} ]; then
  mkdir -p ${out_dir}
fi

# Tune these so you don't oversubscribe CPU/IO:
JOBS=8           # run 8 samples in parallel
SORT_THREADS=32  # threads per samtools sort (DON'T keep 64 if you run multiple jobs)

pids=()

process_sample () {
  local sample="$1"

  in_bam="${in_dir}/${sample}.bam"
  ns_bam="${out_dir}/${sample}.namesort.bam"
  frag="${out_dir}/${sample}.fragments.tsv"

  samtools sort -@ "${SORT_THREADS}" -n -o "${ns_bam}" "${in_bam}"

  python scripts/bam_to_fragments.py \
      --bam "${ns_bam}" \
      --out "${frag}" \
      --cell-tag CB \
      --min-mapq 30 \
      --add-access \
      --ref-fasta ../../data/refdata-cellranger-arc-GRCm39-2024-A/fasta/genome.fa
      
  rm "${ns_bam}"
}

# Launch one subshell per sample (parallel)
for sample in batch1_in1 batch1_in2 batch1_in3 batch1_in4 batch2_in1 batch2_in2 batch2_in3 batch2_in4; do
  (
    process_sample "${sample}"
  ) &
  pids+=("$!")
  # optional: throttle to at most $JOBS concurrent samples
  if (( ${#pids[@]} >= JOBS )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done