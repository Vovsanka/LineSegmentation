script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/../.."
wireframe_src_dir="${project_dir}/../wireframe-dataset/test"
working_state_dir="${project_dir}/experiments-python/wireframe/working-state"

mkdir -p "$working_state_dir"

# for img_path in "${wireframe_src_dir}"/*.jpg; do
#     "${project_dir}/build/LineSegmentation" "$img_path" "$working_state_dir"
# done

img_path="${wireframe_src_dir}/00031546.jpg"
echo "$img_path"
"${project_dir}/build/LineSegmentation" "$img_path" "$working_state_dir"