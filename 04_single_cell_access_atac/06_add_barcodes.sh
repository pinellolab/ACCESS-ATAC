barcode_dir1=../../data/Sitara_scACCESS_mouseAirwayCell_112125/barcodes
barcode_dir2=../../data/Sitara_scACCESS_mouseAirwayCell_121725/barcodes
bam_dir=../../results/04_single_cell_access_atac/01_run_nf/alignments
out_dir=../../results/04_single_cell_access_atac/06_add_barcodes

mkdir -p ${out_dir}

for sample in in1 in2 in3 in4;
do

python scripts/add_barcode_to_bam_v2.py \
--bam_file ${bam_dir}/batch1_${sample}.sorted.bam \
--barcode_file ${barcode_dir1}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_${sample}.fastq.gz \
--corrected_barcode ../../results/04_single_cell_access_atac/03_correct_barcode/barcode.tsv \
--bc_tag CB \
--out_dir ${out_dir} \
--out_name batch1_${sample}

python scripts/add_barcode_to_bam_v2.py \
--bam_file ${bam_dir}/batch2_${sample}.sorted.bam \
--barcode_file ${barcode_dir2}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_${sample}.fastq.gz \
--corrected_barcode ../../results/04_single_cell_access_atac/03_correct_barcode/barcode.tsv \
--bc_tag CB \
--out_dir ${out_dir} \
--out_name batch2_${sample}

samtools index ${out_dir}/batch1_${sample}.bam
samtools index ${out_dir}/batch2_${sample}.bam

done