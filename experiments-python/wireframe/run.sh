script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/../.."
wireframe_src_dir="${project_dir}/../wireframe-dataset/test"
working_state_dir="${project_dir}/experiments-python/wireframe/working-state"

mkdir -p "$working_state_dir"

total=$(ls "${wireframe_src_dir}"/*.jpg 2>/dev/null | wc -l)
count=0

for img_path in "${wireframe_src_dir}"/*.jpg; do
    count=$((count + 1))
    echo "Processing [$count / $total]"
    "${project_dir}/build/LineSegmentation" "$img_path" "$working_state_dir"
done

# img_path="${wireframe_src_dir}/00031546.jpg"
# echo "$img_path"
# "${project_dir}/build/LineSegmentation" "$img_path" "$working_state_dir"