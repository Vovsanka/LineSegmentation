script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"


### 2. YUD (analysis)
yud_src_dir="${project_dir}/../yud-dataset"
yud_out_dir="${experiments_dir}/yud-results-quality"
yud_analysis_out_dir="${experiments_dir}/yud-analysis-quality"
mkdir -p "$yud_analysis_out_dir"
#
total=$(find "$yud_src_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
start_sample=0
sample_count=10
#
count=$start_sample
for img_folder in "${yud_src_dir}"/*/; do
    count=$((count + 1))
    if [ "$count" -lt "$start_sample" ]; then
        continue
    fi
    if [ "$count" -gt "$sample_count" ]; then
        break
    fi
    base=$(basename "$img_folder")
    img_path="${img_folder}${base}.jpg"
    echo ""
    echo "YUD dataset: ${base} [${count} / ${total}]"
    echo ""
    python3 "${experiments_dir}/analyze.py" \
        "${yud_results_out_dir}/working-state-${base}" \
        "${yud_src_dir}/line_mat/${base}_line.mat" \
        "$yud_analysis_out_dir"
done
