import sys
import pandas as pd


def main():
    if len(sys.argv) != 3:
        print("Usage: python aggregate.py <out_dir>")
        sys.exit(1)

    out_dir = sys.argv[1]
    csv_prefix = sys.argv[2]

    prefixes = ["bm_it_", "st_it_", "bm_th_", "st_th_"]
    prefix_names = {
        "bm_it_": "BM-IT",
        "st_it_": "ST-IT",
        "bm_th_": "BM-TH",
        "st_th_": "ST-TH"
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
            df = pd.read_csv(f"{out_dir}/{prefix}{eval_key}_evaluation.csv")
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
            df = pd.read_csv(f"{out_dir}/{prefix}{eval_key}_evaluation.csv")

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


if __name__ == "__main__":
    main()