import os
import sys
import math
import csv

import numpy as np
from scipy.io import loadmat

from line_segment import LineSegment

# read candidates amount 
def read_candidate_amount(candidates_txt: str) -> int:
    with open(candidates_txt, "r") as f:
        n = int(f.readline().strip())
    return n

# read detected line segments
def read_my_line_segments(ls_txt: str) -> list[LineSegment]:
    my_ls: list[LineSegment] = []
    with open(ls_txt, "r") as f:
        n = int(f.readline().strip())
        for _ in range(n):
            x1, y1, x2, y2 = map(float, f.readline().split())
            my_ls.append(LineSegment(x1, y1, x2, y2))
    return my_ls

# read ground truth line segments
def read_gt_line_segments(gt_mat: str) -> list[LineSegment]:
    data = loadmat(gt_mat)
    if "lines" in data:
        arr = data["lines"]
    elif "line" in data:
        arr = data["line"]
    elif "lpos" in data:
        arr = data["lpos"]
    else:
        raise ValueError("No GT line field found in .mat file")

    gt_ls: list[LineSegment] = []
    for x1, y1, x2, y2 in arr:
        gt_ls.append(LineSegment(float(x1), float(y1), float(x2), float(y2)))
    return gt_ls

# angle between the line segments
def angle_diff(ls1: LineSegment, ls2: LineSegment):
    v1 = ls1.direction()
    v2 = ls2.direction()
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = np.clip(cosang, -1, 1)
    return abs(math.degrees(math.acos(cosang)))


# line segment distance (localization error)
def segment_distance(ls1: LineSegment, ls2: LineSegment):
    d1 = ls2.point_to_line_dist(ls1.x1, ls1.y1)
    d2 = ls2.point_to_line_dist(ls1.x2, ls1.y2)
    d3 = ls1.point_to_line_dist(ls2.x1, ls2.y1)
    d4 = ls1.point_to_line_dist(ls2.x2, ls2.y2)
    return (d1 + d2 + d3 + d4) / 4

# project the detected line segment onto the ground truth line segment
def project_onto_gt(det: LineSegment, gt: LineSegment):
    gx, gy = gt.x1, gt.y1
    gdx, gdy = gt.x2 - gt.x1, gt.y2 - gt.y1
    glen2 = gdx * gdx + gdy * gdy

    def proj(x, y):
        t = ((x - gx) * gdx + (y - gy) * gdy) / glen2
        return max(0, min(1, t))

    t1 = proj(det.x1, det.y1)
    t2 = proj(det.x2, det.y2)
    return min(t1, t2), max(t1, t2)

# compute the coverage of the ground truth line segment by projections of the detected line segments
def compute_coverage(gt: LineSegment, dets: list[LineSegment]):
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


# evaluation of the line segment detection
def evaluate_segments(det_ls, gt_ls,
                              angle_thresh,     
                              dist_thresh,       
                              cov_thresh):     

    used_det = set()
    TP = 0
    FN = 0
    localization_errors = []

    for g in gt_ls:
        projections = []

        # 1. Project ALL detections (no filtering yet)
        for i, d in enumerate(det_ls):
            t1, t2 = project_onto_gt(d, g)
            if t2 <= t1:
                continue  # no overlap at all

            ang = angle_diff(g, d)
            dist = segment_distance(g, d)

            projections.append((i, d, t1, t2, ang, dist))

        if not projections:
            FN += 1
            continue

        # 2. Apply relaxed geometric constraints AFTER projection
        valid = [(i, d, dist) for (i, d, t1, t2, ang, dist) in projections
                 if ang < angle_thresh and dist < dist_thresh]

        if not valid:
            FN += 1
            continue

        # 3. Compute coverage using all valid detections
        det_segments = [d for _, d, _ in valid]
        cov = compute_coverage(g, det_segments)

        if cov >= cov_thresh:
            TP += 1
            for idx, d, dist in valid:
                used_det.add(idx)
                localization_errors.append(dist)
        else:
            FN += 1

    FP = len(det_ls) - len(used_det)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    loc_error = np.mean(localization_errors) if localization_errors else 0.0

    return TP, FP, FN, precision, recall, f1, loc_error


def main():
    if len(sys.argv) != 7:
        print("Usage: python eval_segments.py <working_state_dir> <gt_mat> <out_csv> <angle_thresh> <dist_thresh> <cov_thresh>")
        sys.exit(1)

    working_state_dir = sys.argv[1]
    gt_mat = sys.argv[2]
    out_dir = sys.argv[3]

    gt_ls = read_gt_line_segments(gt_mat)

    prefixes: list[str] = ["st_th_", "st_it_", "bm_th_", "bm_it_"]
    strictness: dict[str,tuple[float,float,float]] = {
        "strict": (5.0, 1.0, 0.75),
        "moderate": (10.0, 3.0, 0.75),
        "loose": (20.0, 5.0, 0.5)
    }
    for prefix in prefixes:
        for eval_key in strictness.keys():

        csv_path = os.path.join(out_dir, f"{prefix}{eval_key}_evaluation.csv")

        # Create file with header if it doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Candidates",
                    "GT_count",
                    "Detected_count",
                    "TP",
                    "FP",
                    "FN",
                    "Precision",
                    "Recall",
                    "F1",
                    "LocalizationError"
                ])

    for prefix in prefixes:

        candidates = read_candidate_amount(f"{working_state_dir}/{prefix}candidates.txt")

        for eval_key, eval_config in strictness.items():

            csv_path = os.path.join(out_dir, f"{prefix}{eval_key}_evaluation.csv")

            my_ls = read_my_line_segments(f"{working_state_dir}/{prefix}lines.txt")

            angle_thresh = eval_config[0],
            dist_thresh = eval_config[1],
            cov_thresh = eval_config[2]

            TP, FP, FN, P, R, F1, LE = evaluate_segments(
                my_ls, gt_ls, angle_thresh, dist_thresh, cov_thresh
            )

            csv_path = os.path.join(out_dir, f"{prefix}evaluation.csv")

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    candidates,
                    len(gt_ls),
                    len(my_ls),
                    TP,
                    FP,
                    FN,
                    float(P),
                    float(R),
                    float(F1),
                    float(LE)
                ])

    


if __name__ == "__main__":
    main()
