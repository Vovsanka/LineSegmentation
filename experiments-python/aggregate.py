import sys

import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregate.py <out_csv>")
        sys.exit(1)

    out_dir = sys.argv[1]

    prefixes: list[str] = ["bm_it_", "st_it_", "bm_th_", "st_th_"]
    prefix_names: dict[str,str] = {
        "bm_it_": "BM-IT",
        "st_it_": "ST-IT",
        "bm_th_": "BM-TH",
        "st_th_": "ST-TH"
    }
    strictness: list[str] = ["strict", "moderate", "loose"]
    mean_cols = ["Precision", "Recall", "F1", "LocalizationError"]

    rows = []

    for eval_key in strictness:
        for prefix in prefixes:
            df = pd.read_csv(f"{out_dir}/{prefix}{eval_key}_evaluation.csv")
            mean_vals = df[mean_cols].mean()
            mean_vals["name"] = f"{prefix_names[prefix]} {eval_key.upper()}"
            rows.append(mean_vals)
            
    aggregation = pd.DataFrame(rows).set_index("name")
    aggregation.to_csv(f"{out_dir}/aggregation.csv", index=True)
            

        
if __name__ == "__main__":
    main()
