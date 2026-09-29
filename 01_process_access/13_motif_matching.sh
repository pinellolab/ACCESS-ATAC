in_dir=../../results/01_process_access/10_call_peaks
out_dir=../../results/01_process_access/13_motif_matching

# create output directory if it does not exist
if [ ! -d "$out_dir" ]; then
  mkdir -p "$out_dir"
fi

for sample in HepG2 K562;
do

rgt-motifanalysis matching \
--organism hg38 \
--output-location=${out_dir} \
--input-files ${in_dir}/${sample}_accessatac_peaks.filtered.narrowPeak

done