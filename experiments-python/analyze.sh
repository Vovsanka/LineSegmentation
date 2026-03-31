script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"

analysis_out_dir="${experiments_dir}/analysis-results"

wireframe_src_dir="${project_dir}/../wireframe-dataset"
wireframe_out_dir="${experiments_dir}/wireframe-results"
analysis_wirefame_out_dir="${analysis_out_dir}/analysis-wireframe"

mkdir -p "$analysis_out_dir"

### 1. Wireframe 
total=$(ls "${wireframe_src_dir}/test"/*.jpg 2>/dev/null | wc -l)
count=0
for img_path in "${wireframe_src_dir}/test"/*.jpg; do
    count=$((count + 1))
    ### debug start
    if [ "$count" -gt 2 ]; then
        break
    fi
    ### debug end
    base=$(basename "$img_path" .jpg)
    echo "Wireframe dataset: ${base} [${count} / ${total}]"
    python3 "${experiments_dir}/analyze.py" \
        "${wireframe_out_dir}/working-state-${base}" \
        "${wireframe_src_dir}/line_mat/${base}_line.mat" \
        "${analysis_wireframe_out_dir}/analysis-${base}.csv" \
        "10" "5" "0.5"
done
