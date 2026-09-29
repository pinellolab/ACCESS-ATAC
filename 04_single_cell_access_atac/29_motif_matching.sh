in_dir=../../results/04_single_cell_access_atac/28_get_peaks
out_dir=../../results/04_single_cell_access_atac/29_motif_matching

# create output directory if it does not exist
if [ ! -d "$out_dir" ]; then
  mkdir -p "$out_dir"
fi

rgt-motifanalysis matching \
--organism mm39 \
--output-location=${out_dir} \
--input-files ${in_dir}/peaks.bed