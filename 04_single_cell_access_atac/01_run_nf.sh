in_dir=/data/pinello/PROJECTS/2025_06_ZL_ACCESS/data/refdata-cellranger-arc-GRCm39-2024-A
outdir=/data/pinello/PROJECTS/2025_06_ZL_ACCESS/results/04_single_cell_access_atac/01_run_nf

mkdir -p ${outdir}

nextflow run ../../github/nf-core-accessatacseq/ \
--input ./samples.csv \
--outdir ${outdir} \
--fasta ${in_dir}/fasta/genome.fa \
--fasta_index ${in_dir}/fasta/genome.fa.fai \
--bwameth_index ${in_dir}/bwa_meth_index \
-resume --run_qualimap \
