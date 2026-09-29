in_dir=../../results/04_single_cell_access_atac/31_get_mpbs
out_dir=../../results/04_single_cell_access_atac/32_plot_footprint
bw_dir=../../results/04_single_cell_access_atac/30_fragment_to_bw

# Create output directory if it does not exist
if [ ! -d ${out_dir} ]; then
  mkdir -p ${out_dir}
fi

for sample in Basal Ciliated Club Tuft Hillock;
do

mkdir -p ${out_dir}/${sample}/ATAC
mkdir -p ${out_dir}/${sample}/ACCESS

for tf_mpbs in ${in_dir}/*.bed; do
  python scripts/plot_all_footprint.py \
  --bw_files ${bw_dir}/${sample}_ACCESS.bw \
  --labels ${sample} \
  --bed_file ${tf_mpbs} \
  --extend 250 \
  --out_dir ${out_dir}/${sample}/ACCESS \
  --out_name $(basename ${tf_mpbs} .bed)

  python scripts/plot_all_footprint.py \
  --bw_files ${bw_dir}/${sample}_ATAC.bw \
  --labels ${sample} \
  --bed_file ${tf_mpbs} \
  --extend 250 \
  --out_dir ${out_dir}/${sample}/ATAC \
  --out_name $(basename ${tf_mpbs} .bed)

done
done