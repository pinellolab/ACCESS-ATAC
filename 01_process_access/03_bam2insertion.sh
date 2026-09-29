#!/bin/bash
#SBATCH --job-name=03_bam2insertion
#SBATCH --output=logs/03_bam2insertion.out
#SBATCH --error=logs/03_bam2insertion.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail

conda activate access

module load bedtools/2.31.1

THREADS=32
SORT_MEM=8G                                       # buffer size so --parallel is effective
IN_DIR=../../results/01_process_access/02_filter
OUT_DIR=../../results/01_process_access/03_bam2insertion
CHROM_SIZES=../../data/GRCh38/core.chrom.sizes    # main-chrom sizes (chr1-22, chrX)

mkdir -p ${OUT_DIR} logs

SEED=42

# ---------------------------------------------------------------------------
# Step 1: BAM -> Tn5 insertion BED (one line per insertion, the 5' cut site)
#   - single-end ACCESS : each read gives ONE insertion (its 5' end)
#   - paired-end ATAC   : each mate's 5' end is an insertion, so BOTH mates
#                         contribute -> two insertions per fragment
#   The Tn5 offset (+4 on +, -5 on -) centers the cut site correctly.
# ---------------------------------------------------------------------------
bam_to_insertion_bed () {
    local bam="$1"
    local out_bed="$2"
    bedtools bamtobed -i "${bam}" \
      | awk 'BEGIN{OFS="\t"}
             {
               if ($6 == "+") { start = $2 + 4 }
               else           { start = $3 - 5 }
               if (start < 0) start = 0;
               print $1, start, start+1
             }' \
      | LC_ALL=C sort -k1,1 -k2,2n --parallel=${THREADS} -S ${SORT_MEM} \
      > "${out_bed}"
}

echo "[$(date)] Step 1: generating insertion BEDs"
declare -A INS_BED
declare -A INS_N

for sample in HepG2 K562; do
    access_bam=${IN_DIR}/${sample}.concurrent.accessatac.bam
    atac_bam=${IN_DIR}/${sample}.encode.atacseq.bam

    for assay in accessatac atacseq; do
        if [ "${assay}" = "accessatac" ]; then
            bam=${access_bam}
        else
            bam=${atac_bam}
        fi

        bed=${OUT_DIR}/${sample}.${assay}.insertions.bed
        bam_to_insertion_bed "${bam}" "${bed}"

        n=$(wc -l < "${bed}")
        INS_BED[${sample}_${assay}]=${bed}
        INS_N[${sample}_${assay}]=${n}
        echo "  [${sample} ${assay}] insertions = ${n}"
    done
done

# ---------------------------------------------------------------------------
# Step 2: per cell line, downsample both assays to the SAME insertion count
#         (the smaller of the two), reproducible via fixed seed
# ---------------------------------------------------------------------------
echo "[$(date)] Step 2: downsampling to matched insertion counts"

for sample in HepG2 K562; do
    n_access=${INS_N[${sample}_accessatac]}
    n_atac=${INS_N[${sample}_atacseq]}
    if [ "${n_access}" -le "${n_atac}" ]; then target=${n_access}; else target=${n_atac}; fi
    echo "  [${sample}] ACCESS=${n_access} ATAC=${n_atac} -> target=${target}"

    for assay in accessatac atacseq; do
        bed=${INS_BED[${sample}_${assay}]}
        n=${INS_N[${sample}_${assay}]}
        ds_bed=${OUT_DIR}/${sample}.${assay}.insertions.ds.bed

        if [ "${n}" -eq "${target}" ]; then
            cp "${bed}" "${ds_bed}"
        else
            # subsample exactly 'target' lines, then re-sort (shuf breaks order)
            shuf --random-source=<(yes ${SEED}) -n ${target} "${bed}" \
              | LC_ALL=C sort -k1,1 -k2,2n --parallel=${THREADS} -S ${SORT_MEM} \
              > "${ds_bed}"
        fi
        echo "    [${sample} ${assay}] after downsample = $(wc -l < ${ds_bed})"
    done
done

# ---------------------------------------------------------------------------
# Step 3: downsampled insertion BED -> CPM-normalized bigWig
# ---------------------------------------------------------------------------
echo "[$(date)] Step 3: BED -> bigWig"

for sample in HepG2 K562; do
    for assay in accessatac atacseq; do
        ds_bed=${OUT_DIR}/${sample}.${assay}.insertions.ds.bed
        bg=${OUT_DIR}/${sample}.${assay}.ds.bedgraph
        bw=${OUT_DIR}/${sample}.${assay}.ds.bw

        # scale each insertion to CPM (1e6 / total insertions) so tracks compare
        n=$(wc -l < "${ds_bed}")
        scale=$(awk -v n=${n} 'BEGIN{printf "%.8f", 1000000/n}')

        bedtools genomecov -i "${ds_bed}" -g "${CHROM_SIZES}" -bg -scale ${scale} \
          | LC_ALL=C sort -k1,1 -k2,2n --parallel=${THREADS} -S ${SORT_MEM} \
          > "${bg}"
        bedGraphToBigWig "${bg}" "${CHROM_SIZES}" "${bw}"
        rm "${bg}"
        echo "  [${sample} ${assay}] wrote ${bw} (${n} insertions, scale=${scale})"
    done
done

echo "[$(date)] done"