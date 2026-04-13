script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"
executable="${project_dir}/build/LineSegmentation"


#### 1. Wireframe (clustering)
wireframe_src_dir="${project_dir}/../wireframe-dataset"
wireframe_out_dir="${experiments_dir}/wireframe-results-quality"
#
total=$(ls "${wireframe_src_dir}/test"/*.jpg 2>/dev/null | wc -l)
start_sample=1
sample_count=15
#
count=0
for img_path in "${wireframe_src_dir}/test"/*.jpg; do
    count=$((count + 1))
    if [ "$count" -lt "$start_sample" ]; then
        continue
    fi
    if [ "$count" -gt "$sample_count" ]; then
        break
    fi
    base=$(basename "$img_path" .jpg)
    echo ""
    echo "Wireframe dataset: ${base} [${count} / ${total}]"
    echo ""
    working_state_dir="${wireframe_out_dir}/working-state-${base}"
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--th" "--on-cg"
    "$executable" "$img_path" "$working_state_dir" "--st" "--it" "--on-cg" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--th" "--on-cg" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cg" 
    # 
    "$executable" "$img_path" "$working_state_dir" "--st" "--th" "--on-cl" "--on-el" "--ga-kl"
    "$executable" "$img_path" "$working_state_dir" "--st" "--it" "--on-cl" "--on-el" "--ga-kl"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--th" "--on-cl" "--on-el" "--ga-kl"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--mws"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--ga"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--kl"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--ga-kl"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--mws-kl"
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
done
