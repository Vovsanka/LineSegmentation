import sys
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python aggregate.py <out_dir> <csv_prefix> <clustering_flag>")
        sys.exit(1)

    out_dir = sys.argv[1]
    csv_prefix = sys.argv[2]
    clustering_flag = None
    if len(sys.argv) >= 4:
        clustering_flag = sys.argv[3]
    clustering_prefix = ""
    if clustering_flag is not None:
        clustering_prefix = f"_{clustering_flag}"

    prefixes = ["bm_it", "st_it", "bm_th", "st_th"]
    prefix_names = {
        "bm_it": "BM-IT",
        "st_it": "ST-IT",
        "bm_th": "BM-TH",
        "st_th": "ST-TH"
    }

    strictness_levels = ["strict", "moderate", "loose"]
    strictness_map = {
        "strict": "STR",
        "moderate": "MOD",
        "loose": "LOO"
    }

    metrics = {
        "Precision": f"{csv_prefix}-precision.csv",
        "Recall": f"{csv_prefix}-recall.csv",
        "F1": f"{csv_prefix}-f1.csv",
        "LocalizationError": f"{csv_prefix}-localization.csv"
    }

    # store results per metric
    results = {m: [] for m in metrics}

    for eval_key in strictness_levels:
        for prefix in prefixes:
            df = pd.read_csv(f"{out_dir}/{prefix}_{eval_key}{clustering_prefix}_evaluation.csv")
            mean_vals = df.mean(numeric_only=True)

            method = prefix_names[prefix]
            
            strict = strictness_map[eval_key]

            for metric in metrics:
                results[metric].append({
                    "strictness": strict,
                    "method": method,
                    "value": mean_vals[metric]
                })

    # convert each metric into wide CSV
    for metric, filename in metrics.items():
        df = pd.DataFrame(results[metric])

        pivot = df.pivot(index="strictness", columns="method", values="value")

        # Ensure correct order
        pivot = pivot.reindex(["STR", "MOD", "LOO"])

        # Save
        pivot.to_csv(f"{out_dir}/{filename}")

    # create aggregated csv with all mean values
    mean_cols = ["Candidates","GT_count","Detected_count","TP","FP","FN","Precision","Recall","F1","LocalizationError"]
    
    rows = []

    for eval_key in strictness_levels:
        for prefix in prefixes:
            df = pd.read_csv(f"{out_dir}/{prefix}_{eval_key}{clustering_prefix}_evaluation.csv")

            mean_vals = df[mean_cols].mean()

            method = prefix_names[prefix]
            strict = strictness_map[eval_key]

            mean_vals["name"] = f"{method} {strict}"

            rows.append(mean_vals)
            
    aggregation = pd.DataFrame(rows).set_index("name")
    aggregation["Precision"] *= 100
    aggregation["Recall"] *= 100
    aggregation["F1"] *= 100
    aggregation = aggregation.round(1)
    aggregation["Candidates"] = aggregation["Candidates"].round(0).astype(int)
    aggregation.to_csv(f"{out_dir}/{csv_prefix}-aggregation.csv")

    ####
    clustering_methods = ["MWS", "GA", "KL", "GA+KL", "MWS+KL"]
    mean_cols = ["Precision", "Recall", "F1"]
    rows = []
    for clustering in clustering_methods:
        df = pd.read_csv(f"{out_dir}/bm_it_{clustering}_loose_clustering.csv")
        mean_vals = df[mean_cols].mean()
        mean_vals["Clustering"] = clustering
        rows.append(mean_vals)
    aggregation = pd.DataFrame(rows).set_index("Clustering")
    aggregation["Precision"] *= 100
    aggregation["Recall"] *= 100
    aggregation["F1"] *= 100
    aggregation = aggregation.round(1)
    aggregation.to_csv(f"{out_dir}/{csv_prefix}-clustering.csv")

    ####
    stages = [
        "st_lp", "st_tc", "st_ic",
        "bm_lp", "bm_tc", "bm_ic",
        "st_th_cg", "st_it_cg", "bm_th_cg", "bm_it_cg",
        "st_th_GA+KL_cl", "st_th_GA+KL_el", "st_th_MWS_cl", "st_th_MWS_el",
        "st_it_GA+KL_cl", "st_it_GA+KL_el", "st_it_MWS_cl", "st_it_MWS_el",
        "bm_th_GA+KL_cl", "bm_th_GA+KL_el", "bm_th_MWS_cl", "bm_th_MWS_el",
        "bm_it_GA+KL_cl", "bm_it_GA+KL_el", "bm_it_MWS_cl", "bm_it_MWS_el",
        "bm_it_GA_cl", "bm_it_GA_el", "bm_it_KL_cl", "bm_it_KL_el",
        "bm_it_MWS+KL_cl", "bm_it_MWS+KL_el"
    ]
    df = pd.read_csv(f"{out_dir}/time.csv")
    mean_vals = df[stages].mean()
    aggregation = pd.DataFrame([mean_vals])
    aggregation = aggregation.round(3)
    aggregation.to_csv(f"{out_dir}/{csv_prefix}-time.csv", index=False)


if __name__ == "__main__":
    main()