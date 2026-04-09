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

# read detected line segments (YUD+)
def read_yud_plus_line_segments(gt_txt: str) -> list[LineSegment]:
    segments: list[LineSegment] = []

    with open(gt_txt, "r") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            
            parts = line.strip().split()

            # extract endpoints (homogeneous coordinates)
            x1 = float(parts[6])
            y1 = float(parts[7])
            z1 = float(parts[8])
            x2 = float(parts[9])
            y2 = float(parts[10])
            z2 = float(parts[11])
            
            # convert from homogeneous if needed
            if z1 != 0:
                x1 /= z1
                y1 /= z1
            if z2 != 0:
                x2 /= z2
                y2 /= z2
            
            segments.append(LineSegment(x1, y1, x2, y2))
    
    return segments


# read ground truth line segments (wireframe)
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
    if len(sys.argv) < 5:
        print("Usage: python analyze.py <working_state_dir> <gt_file> <out_dir> <dataset_flag> <clustering_flag>")
        sys.exit(1)

    working_state_dir = sys.argv[1]
    gt_file = sys.argv[2]
    out_dir = sys.argv[3]   
    if sys.argv[4] == "--wireframe":
        dataset_flag = "Wireframe"
    elif sys.argv[4] == "--yud":
        dataset_flag = "YUD+"
    else:
        raise NameError("A wrong dataset flag!")
    if dataset_flag == "Wireframe":
        gt_ls = read_wireframe_line_segments(gt_file)
    else:
        gt_ls = read_yud_plus_line_segments(gt_file)
    clustering_flag = None
    if len(sys.argv) >= 6:
        clustering_flag = sys.argv[5]
    clustering_prefix = ""
    if clustering_flag is not None:
        clustering_prefix = f"_{clustering_flag}" 

    prefixes: list[str] = ["bm_it", "st_it", "st_th", "bm_th"]
    strictness: dict[str,tuple[float,float,float]] = {
        "strict": (5.0, 1.0, 0.75),
        "moderate": (10.0, 3.0, 0.75),
        "loose": (20.0, 5.0, 0.5)
    }
    for prefix in prefixes:
        for eval_key in strictness.keys():

            csv_path = os.path.join(out_dir, f"{prefix}_{eval_key}{clustering_prefix}_evaluation.csv")

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

        candidates = read_candidate_amount(f"{working_state_dir}/{prefix}_candidates.txt")

        for eval_key, eval_config in strictness.items():

            csv_path = os.path.join(out_dir, f"{prefix}_{eval_key}{clustering_prefix}_evaluation.csv")

            my_ls = read_my_line_segments(f"{working_state_dir}/{prefix}{clustering_prefix}_lines.txt")

            angle_thresh = eval_config[0]
            dist_thresh = eval_config[1]
            cov_thresh = eval_config[2]

            TP, FP, FN, P, R, F1, LE = evaluate_segments(
                my_ls, gt_ls, angle_thresh, dist_thresh, cov_thresh
            )

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
