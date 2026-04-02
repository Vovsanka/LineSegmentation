script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"


### 1. Wireframe (analysis)
wireframe_src_dir="${project_dir}/../wireframe-dataset"
wireframe_results_out_dir="${experiments_dir}/wireframe-results-quality"
wireframe_analysis_out_dir="${experiments_dir}/wireframe-analysis-quality"
rm -rf "$wireframe_analysis_out_dir"
mkdir -p "$wireframe_analysis_out_dir"
#
total=$(ls "${wireframe_src_dir}/test"/*.jpg 2>/dev/null | wc -l)
start_sample=0
sample_count=20
#
count=$start_sample
count=0
for img_path in "${wireframe_src_dir}/test"/*.jpg; do
    count=$((count + 1))
    if [ "$count" -lt "$start_sample" ]; then
        continue
    fi
    if [ "$count" -gt "$sample_count" ]; then
        break
    fi
    #
    base=$(basename "$img_path" .jpg)
    echo ""
    echo "Wireframe dataset: ${base} [${count} / ${total}]"
    echo ""
    python3 "${experiments_dir}/analyze.py" \
        "${wireframe_results_out_dir}/working-state-${base}" \
        "${wireframe_src_dir}/line_mat/${base}_line.mat" \
        "$wireframe_analysis_out_dir" \
        "--wireframe"
done
