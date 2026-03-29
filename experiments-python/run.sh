script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"
executable="${project_dir}/build/LineSegmentation"

wireframe_src_dir="${project_dir}/../wireframe-dataset"

### 1 Wireframe
wireframe_out_dir="${experiments_dir}/wireframe-results"
mkdir -p "$wireframe_out_dir"
total=$(ls "${wireframe_src_dir}/test"/*.jpg 2>/dev/null | wc -l)
count=0
for img_path in "${wireframe_src_dir}/test"/*.jpg; do
    count=$((count + 1))
    ### debug start
    if [ "$count" -gt 3 ]; then
        break
    fi
    ### debug end
    base=$(basename "$img_path" .jpg)
    echo "Wireframe dataset: ${base} [${count} / ${total}]"
    working_state_dir="${wireframe_out_dir}/working-state-${base}"
    mkdir -p "$working_state_dir"
    "$executable" "$img_path" "$working_state_dir" "--on-show" "--off-lp" "--off-tc" "--off-ic" "--off-cg" "--off-cl" "--off-el"
done

