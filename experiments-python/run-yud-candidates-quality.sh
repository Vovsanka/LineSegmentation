script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"
executable="${project_dir}/build/LineSegmentation"


#### 2. YUD (candidates)
yud_src_dir="${project_dir}/../yud-dataset"
yud_out_dir="${experiments_dir}/yud-results-quality"
mkdir -p "$yud_out_dir"
#
total=$(find "$yud_src_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
total=$((total - 1))
start_sample=0
sample_count=1
#
count=$start_sample
for img_folder in "${yud_src_dir}"/*/; do
    base=$(basename "$img_folder")
    if [ "$base" = "lines" ]; then
        continue
    fi
    count=$((count + 1))
    if [ "$count" -lt "$start_sample" ]; then
        continue
    fi
    if [ "$count" -gt "$sample_count" ]; then
        break
    fi
    img_path="${img_folder}${base}.jpg"
    echo ""
    echo "YUD dataset: ${base} [${count} / ${total}]"
    echo ""
    working_state_dir="${yud_out_dir}/working-state-${base}"
    mkdir -p "$working_state_dir"
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--on-lp" "--on-tc" "--on-ic" 
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--on-lp" "--on-tc" "--on-ic" 
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
done

