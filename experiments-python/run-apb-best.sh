script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"
executable="${project_dir}/build/LineSegmentation"


#### 3. APB
apb_src_dir="${project_dir}/../apb-dataset"
apb_out_dir="${experiments_dir}/apb-results"
mkdir -p "$apb_out_dir"
#
total=$(ls "${apb_src_dir}"/*.jpg 2>/dev/null | wc -l)
start_sample=1
sample_count=6
#
count=0
for img_path in "${apb_src_dir}/test"/*.jpg; do
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
    echo "APB dataset: ${base} [${count} / ${total}]"
    echo ""
    working_state_dir="${apb_out_dir}/working-state-${base}"
    mkdir -p "$working_state_dir"
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--on-lp" "--on-tc" "--on-ic" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--on-lp" "--on-tc" "--on-ic" 
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--th" "--on-cg"
    "$executable" "$img_path" "$working_state_dir" "--st" "--it" "--on-cg" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--th" "--on-cg" 
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cg" 
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--th" "--on-cl" "--on-el"
    "$executable" "$img_path" "$working_state_dir" "--st" "--it" "--on-cl" "--on-el"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--th" "--on-cl" "--on-el"
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-cl" "--on-el"
    #
    "$executable" "$img_path" "$working_state_dir" "--st" "--th" "--on-show"  
    "$executable" "$img_path" "$working_state_dir" "--st" "--it" "--on-show"  
    "$executable" "$img_path" "$working_state_dir" "--bm" "--th" "--on-show"  
    "$executable" "$img_path" "$working_state_dir" "--bm" "--it" "--on-show"  
done

