script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="${script_dir}/.."
experiments_dir="${project_dir}/experiments-python"

analysis_out_dir="${experiments_dir}/analysis-results"

wireframe_src_dir="${project_dir}/../wireframe-dataset"
wireframe_out_dir="${experiments_dir}/wireframe-results"

mkdir -p "$analysis_out_dir"

### 1 Wireframe 

for ls_txt in "${wireframe_out_dir}"/*.txt; do
    base=$(basename "$ls_txt" .txt)
    
    python3 "${experiments_dir}/analyze.py" \
        "$ls_txt" \
        "${wireframe_src_dir}/line_mat/${base}_line.mat" \
        "${analysis_out_dir}/wireframe.csv"
done

