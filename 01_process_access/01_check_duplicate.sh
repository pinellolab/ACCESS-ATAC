#!/bin/bash
#SBATCH --job-name=01_check_duplicate
#SBATCH --output=logs/01_check_duplicate.out
#SBATCH --error=logs/01_check_duplicate.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

# --- environment ---
conda activate access          # adjust if your samtools env has another name

THREADS=64
IN_BAM=../../data/Concurrent_ACCESS_ATAC/062424_HepG2_concurrent_ACCESS-ATAC.dedup.hg38_core.bam
OUT_DIR=../../results/01_process_access/01_check_duplicate

mkdir -p ${OUT_DIR}

echo "[$(date)] input: ${IN_BAM}"
echo "[$(date)] output dir: ${OUT_DIR}"

# --- 1. mark duplicates by coordinate (single-end: 5' start + strand) ---
echo "[$(date)] --- samtools markdup (coordinate-level) ---"
samtools markdup -@ "${THREADS}" -s \
  "${IN_BAM}" "${OUT_DIR}/marked.bam" \
  2> "${OUT_DIR}/01_markdup_stats.txt"
samtools index -@ "${THREADS}" "${OUT_DIR}/marked.bam"
cat "${OUT_DIR}/01_markdup_stats.txt"
 
# # --- 2. duplicate rate summary ---
# echo "[$(date)] --- duplicate rate ---"
# TOTAL=$(samtools view -c -F 0x900 "${OUT_DIR}/marked.bam")
# DUPS=$(samtools view -c -F 0x900 -f 0x400 "${OUT_DIR}/marked.bam")
# awk -v t="${TOTAL}" -v d="${DUPS}" 'BEGIN{
#   printf "  primary reads            : %d\n", t;
#   printf "  coord-flagged duplicates : %d\n", d;
#   printf "  extra duplicate rate     : %.2f%%\n", (t>0? 100*d/t : 0);
#   print "";
#   if (t>0 && 100*d/t < 2)
#     print "  -> LOW: cz-dedup already removed nearly all duplicates; coordinate dedup adds little.";
#   else if (t>0 && 100*d/t < 10)
#     print "  -> MODERATE: some duplicates missed by sequence dedup; inspect distribution.";
#   else
#     print "  -> HIGH: substantial extra duplicates OR single-end over-collapsing; inspect distribution.";
# }' | tee "${OUT_DIR}/02_dup_rate.txt"
 
# # --- 3. separate duplicate vs non-duplicate, make bigWigs to inspect location ---
# echo "[$(date)] --- separating duplicate vs non-duplicate reads ---"
# samtools view -@ "${THREADS}" -b -f 0x400 -F 0x900 \
#   "${OUT_DIR}/marked.bam" > "${OUT_DIR}/dups_only.bam"
# samtools view -@ "${THREADS}" -b -F 0x500 \
#   "${OUT_DIR}/marked.bam" > "${OUT_DIR}/nodup.bam"
# samtools index "${OUT_DIR}/dups_only.bam"
# samtools index "${OUT_DIR}/nodup.bam"
# echo "  dups_only reads : $(samtools view -c ${OUT_DIR}/dups_only.bam)"
# echo "  nodup reads     : $(samtools view -c ${OUT_DIR}/nodup.bam)"
 
# if command -v bamCoverage >/dev/null 2>&1; then
#   echo "[$(date)] --- bigWigs (deeptools) ---"
#   bamCoverage -b "${OUT_DIR}/dups_only.bam" -o "${OUT_DIR}/dups_only.bw" \
#     -p "${THREADS}" --normalizeUsing CPM 2>/dev/null || echo "  (dups bigWig failed)"
#   bamCoverage -b "${OUT_DIR}/nodup.bam" -o "${OUT_DIR}/nodup.bw" \
#     -p "${THREADS}" --normalizeUsing CPM 2>/dev/null || echo "  (nodup bigWig failed)"
# else
#   echo "  deeptools not found; skipping bigWig."
# fi
 
# # --- 4. duplicate counts per chromosome ---
# echo "[$(date)] --- per-chromosome duplicate counts ---"
# samtools view -F 0x900 -f 0x400 "${OUT_DIR}/marked.bam" \
#   | cut -f3 | sort | uniq -c | sort -k2,2V \
#   > "${OUT_DIR}/04_dup_per_chrom.txt"
# head "${OUT_DIR}/04_dup_per_chrom.txt"
 
# echo "[$(date)] done. Key files in ${OUT_DIR}/:"
# echo "    01_markdup_stats.txt  02_dup_rate.txt  04_dup_per_chrom.txt"
# echo "    dups_only.bw / nodup.bw (load in IGV)"