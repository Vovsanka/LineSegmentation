script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"
executable="${project_dir}/build/LineSegmentation"


#### 2. YUD (clustering)
yud_src_dir="${project_dir}/../yud-dataset"
yud_out_dir="${experiments_dir}/yud-results-quality"
#
total=$(find "$yud_src_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
total=$((total - 1))
start_sample=1
sample_count=10
#
count=0
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
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--th" "--on-cg" "--on-cl" "--on-el"
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
    "$executable" "$img_path" "$working_state_dir" "--st" "--it" "--on-cg" "--on-cl" "--on-el"
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--th" "--on-cg" "--on-cl" "--on-el"
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cg" "--on-cl" "--on-el"
    # "$executable" "$img_path" "$working_state_dir" "--on-show" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--mws"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--ga"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--kl"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--ga-kl"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el" "--mws-kl"
done
