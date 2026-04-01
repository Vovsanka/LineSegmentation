script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"
executable="${project_dir}/build/LineSegmentation"


#### 1. Wireframe (candidates)
wireframe_src_dir="${project_dir}/../wireframe-dataset"
wireframe_out_dir="${experiments_dir}/wireframe-results-quality"
mkdir -p "$wireframe_out_dir"
#
total=$(ls "${wireframe_src_dir}/test"/*.jpg 2>/dev/null | wc -l)
start_sample=0
sample_count=20
#
count=$start_sample
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
    working_state_dir="${wireframe_out_dir}/working-state-${base}"
    mkdir -p "$working_state_dir"
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--on-lp" "--on-tc" "--on-ic" 
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--on-lp" "--on-tc" "--on-ic" 
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
done

