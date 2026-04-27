import os
import sys
import math
import csv

import numpy as np
from scipy.io import loadmat

from line_segment import LineSegment


def read_candidate_amount(candidates_txt: str) -> int:
    with open(candidates_txt, "r") as f:
        return int(f.readline().strip())


def read_my_line_segments(ls_txt: str) -> list[LineSegment]:
    segs = []
    try:
        with open(ls_txt, "r") as f:
            n = int(f.readline().strip())
            for _ in range(n):
                x1, y1, x2, y2 = map(float, f.readline().split())
                segs.append(LineSegment(x1, y1, x2, y2))
    except:
        print(f"'{ls_txt}' file not found")
    return segs


def read_yud_plus_line_segments(gt_txt: str) -> list[LineSegment]:
    segs = []
    with open(gt_txt, "r") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split()
            x1, y1, z1 = map(float, parts[6:9])
            x2, y2, z2 = map(float, parts[9:12])
            if z1 != 0:
                x1 /= z1
                y1 /= z1
            if z2 != 0:
                x2 /= z2
                y2 /= z2
            segs.append(LineSegment(x1, y1, x2, y2))
    return segs


def read_wireframe_line_segments(gt_mat: str) -> list[LineSegment]:
    data = loadmat(gt_mat)
    if "lines" in data:
        arr = data["lines"]
    elif "line" in data:
        arr = data["line"]
    elif "lpos" in data:
        arr = data["lpos"]
    else:
        raise ValueError("No GT line field found in .mat file")

    segs = []
    for x1, y1, x2, y2 in arr:
        segs.append(LineSegment(float(x1), float(y1), float(x2), float(y2)))
    return segs

def read_time_data(time_logs: str) -> dict[str, float]:
    time_data = {}
    with open(time_logs, "r") as tl:
        for line in tl:
            line = line.strip()
            if not line:
                continue
            stage_name, stage_time = line.split()
            time_data[stage_name] = float(stage_time)
    return time_data 

def angle_diff(ls1: LineSegment, ls2: LineSegment) -> float:
    v1 = ls1.direction()
    v2 = ls2.direction()
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = np.clip(cosang, -1, 1)
    return abs(math.degrees(math.acos(cosang)))


def distance_threshold_original(gt: LineSegment, det: LineSegment) -> float:
    d1 = gt.point_to_line_dist(det.x1, det.y1)
    d2 = gt.point_to_line_dist(det.x2, det.y2)
    return max(d1, d2)

def localization_error_original(gt: LineSegment, det: LineSegment) -> float:
    d1 = gt.point_to_line_dist(det.x1, det.y1)
    d2 = gt.point_to_line_dist(det.x2, det.y2)
    return 0.5 * (d1 + d2)

def project_onto_gt(det: LineSegment, gt: LineSegment):
    gx, gy = gt.x1, gt.y1
    gdx, gdy = gt.x2 - gt.x1, gt.y2 - gt.y1
    glen2 = gdx * gdx + gdy * gdy

    if glen2 < 1e-12:
        return 0.0, 0.0

    def proj(x, y):
        t = ((x - gx) * gdx + (y - gy) * gdy) / glen2
        return max(0.0, min(1.0, t))

    t1 = proj(det.x1, det.y1)
    t2 = proj(det.x2, det.y2)
    return min(t1, t2), max(t1, t2)


def compute_coverage(gt: LineSegment, dets: list[LineSegment]) -> float:
    intervals = []
    for d in dets:
        t1, t2 = project_onto_gt(d, gt)
        if t2 > t1:
            intervals.append((t1, t2))

    if not intervals:
        return 0.0

    intervals.sort()
    merged = []
    cur_s, cur_e = intervals[0]

    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return sum(e - s for s, e in merged)


def evaluate_segments(det_ls, gt_ls, angle_thresh, dist_thresh, cov_thresh):
    used_det = set()
    TP = 0
    FN = 0
    loc_errors = []

    for g in gt_ls:

        compatible = []
        for i, d in enumerate(det_ls):
            if angle_diff(g, d) < angle_thresh:
                if distance_threshold_original(g, d) < dist_thresh:
                    compatible.append((i, d))

        if not compatible:
            FN += 1
            continue

        det_segments = [d for _, d in compatible]
        cov = compute_coverage(g, det_segments)

        if cov >= cov_thresh:
            TP += 1

            best_err = min(localization_error_original(g, d) for _, d in compatible)
            loc_errors.append(best_err)

            for idx, _ in compatible:
                used_det.add(idx)

        else:
            FN += 1

    FP = len(det_ls) - len(used_det)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    loc_err = float(np.mean(loc_errors)) if loc_errors else 0.0

    return TP, FP, FN, precision, recall, f1, loc_err


def main():
    if len(sys.argv) < 5:
        print("Usage: python analyze.py <working_state_dir> <gt_file> <out_dir> <dataset_flag> <clustering_flag>")
        sys.exit(1)

    working_state_dir = sys.argv[1]
    gt_file = sys.argv[2]
    out_dir = sys.argv[3]

    if sys.argv[4] == "--wireframe":
        gt_ls = read_wireframe_line_segments(gt_file)
    elif sys.argv[4] == "--yud":
        gt_ls = read_yud_plus_line_segments(gt_file)
    else:
        raise NameError("Invalid dataset flag")

    clustering_flag = sys.argv[5] if len(sys.argv) >= 6 else None
    clustering_prefix = f"_{clustering_flag}" if clustering_flag else ""

    prefixes = ["bm_it", "st_it", "st_th", "bm_th"]

    strictness = {
        "strict":   (5.0, 1.0, 0.75),
        "moderate": (10.0, 3.0, 0.75),
        "loose":    (20.0, 5.0, 0.5),
    }

    # Prepare CSVs
    for prefix in prefixes:
        for key in strictness:
            csv_path = os.path.join(out_dir, f"{prefix}_{key}{clustering_prefix}_evaluation.csv")
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Candidates", "GT_count", "Detected_count",
                        "TP", "FP", "FN",
                        "Precision", "Recall", "F1",
                        "LocalizationError"
                    ])

    # Run evaluation
    for prefix in prefixes:
        candidates = read_candidate_amount(f"{working_state_dir}/{prefix}_candidates.txt")

        for key, (ang_t, dist_t, cov_t) in strictness.items():
            csv_path = os.path.join(out_dir, f"{prefix}_{key}{clustering_prefix}_evaluation.csv")

            det_ls = read_my_line_segments(
                f"{working_state_dir}/{prefix}{clustering_prefix}_lines.txt"
            )

            TP, FP, FN, P, R, F1, LE = evaluate_segments(
                det_ls, gt_ls, ang_t, dist_t, cov_t
            )

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    candidates,
                    len(gt_ls),
                    len(det_ls),
                    TP, FP, FN,
                    float(P), float(R), float(F1),
                    float(LE)
                ])
    
    # BM-IT LOO clustering
    clustering_methods = ["MWS", "GA", "KL", "GA+KL", "MWS+KL"]
    for clustering in clustering_methods:
        csv_path = os.path.join(out_dir, f"bm_it_{clustering}_loose_clustering.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Precision", "Recall", "F1",
                    ])
    for clustering in clustering_methods:
        csv_path = os.path.join(out_dir, f"bm_it_{clustering}_loose_clustering.csv")
        det_ls = read_my_line_segments(
            f"{working_state_dir}/bm_it_{clustering}_lines.txt"
        )

        TP, FP, FN, P, R, F1, LE = evaluate_segments(
            det_ls, gt_ls, ang_t, dist_t, cov_t
        )

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                float(P), float(R), float(F1)
            ])

    ### COMPUTATION TIME from time_logs
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
    time_data = read_time_data(f"{working_state_dir}/time-logs.txt")
    csv_path = os.path.join(out_dir, "time.csv")
    if not os.path.exists(csv_path):
       with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(stages) 
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        row_data = []
        for stage_name in stages:
            if stage_name in time_data:
                row_data.append(time_data[stage_name])
            else:
                row_data.append(0)
        writer.writerow(row_data)



if __name__ == "__main__":
    main()
