script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"

working_state_dir="${experiments_dir}/working-state"

wireframe_src_dir="${project_dir}/../wireframe-dataset"
wireframe_out_dir="${experiments_dir}/wireframe-results"

mkdir -p "$working_state_dir"
mkdir -p "$wireframe_out_dir"

### 1 Wireframe

total=$(ls "${wireframe_src_dir}/test"/*.jpg 2>/dev/null | wc -l)
count=0

for img_path in "${wireframe_src_dir}/test"/*.jpg; do
    count=$((count + 1))
    # TODO: remove this stop after 5 images
    if [ "$count" -gt 3 ]; then
        break
    fi
    echo "Wireframe dataset [$count / $total]"
    "${project_dir}/build/LineSegmentation" "$img_path" "$working_state_dir"
    # extract resulting line segments
    base=$(basename "$img_path" .jpg)
    src="${working_state_dir}/or_lines.txt"
    dst="${wireframe_out_dir}/${base}.txt"
    mv "$src" "$dst"
done

