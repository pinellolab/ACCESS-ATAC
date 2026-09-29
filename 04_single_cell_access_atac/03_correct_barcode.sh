in_dir1=../../data/Sitara_scACCESS_mouseAirwayCell_112125/barcodes
in_dir2=../../data/Sitara_scACCESS_mouseAirwayCell_121725/barcodes
out_dir=../../results/04_single_cell_access_atac/03_correct_barcode

mkdir -p ${out_dir}

python scripts/barcode_correction.py \
${in_dir1}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_in1.fastq.gz \
${in_dir1}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_in2.fastq.gz \
${in_dir1}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_in3.fastq.gz \
${in_dir1}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_in4.fastq.gz \
${in_dir2}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_in1.fastq.gz \
${in_dir2}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_in2.fastq.gz \
${in_dir2}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_in3.fastq.gz \
${in_dir2}/Sitara_1SR_SingleCellmouse_ATAC-ACCESS_10162025_in4.fastq.gz \
../../data/10xGenomics/737K-cratac-v1.txt \
--output ${out_dir}/barcode.tsv