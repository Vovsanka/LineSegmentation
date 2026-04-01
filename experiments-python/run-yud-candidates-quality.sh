script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"
executable="${project_dir}/build/LineSegmentation"


#### 1. YUD (candidates)
yud_src_dir="${project_dir}/../yud-dataset"
yud_out_dir="${experiments_dir}/yud-results-quality"
mkdir -p "$yud_out_dir"
#
total=$(find "$yud_src_dir/test" -mindepth 1 -maxdepth 1 -type d | wc -l)
start_sample=0
sample_count=20
#
count=$start_sample
for img_folder in "${yud_src_dir}/*/"; do
    count=$((count + 1))
    if [ "$count" -lt "$start_sample" ]; then
        continue
    fi
    if [ "$count" -gt "$sample_count" ]; then
        break
    fi
    #
    base=$(basename "$img_folder")
    img_path="${img_folder}/${base}.jpg"
    echo "YUD dataset: ${base} [${count} / ${total}]"
    working_state_dir="${yud_out_dir}/working-state-${base}"
    mkdir -p "$working_state_dir"
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--on-lp" "--on-tc" "--on-ic" 
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--on-lp" "--on-tc" "--on-ic" 
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
done

