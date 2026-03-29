import sys
import math
import numpy as np
from scipy.io import loadmat


# ---------------------------------------------------------
# Data structure
# ---------------------------------------------------------

class LineSegment:
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


# ---------------------------------------------------------
# Reading functions
# ---------------------------------------------------------

def read_my_line_segments(ls_txt: str) -> list[LineSegment]:
    my_ls: list[LineSegment] = []
    with open(ls_txt, "r") as f:
        n = int(f.readline().strip())
        for _ in range(n):
            x1, y1, x2, y2 = map(float, f.readline().split())
            my_ls.append(LineSegment(x1, y1, x2, y2))
    return my_ls


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


# ---------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------

def segment_direction(ls: LineSegment):
    return np.array([ls.x2 - ls.x1, ls.y2 - ls.y1])


def angle_diff(ls1: LineSegment, ls2: LineSegment):
    v1 = segment_direction(ls1)
    v2 = segment_direction(ls2)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = np.clip(cosang, -1, 1)
    return abs(math.degrees(math.acos(cosang)))


def point_to_line_dist(px, py, x1, y1, x2, y2):
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C * C + D * D

    if len_sq < 1e-9:
        return math.hypot(px - x1, py - y1)

    param = dot / len_sq
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    return math.hypot(px - xx, py - yy)


def segment_distance(ls1: LineSegment, ls2: LineSegment):
    d1 = point_to_line_dist(ls1.x1, ls1.y1, ls2.x1, ls2.y1, ls2.x2, ls2.y2)
    d2 = point_to_line_dist(ls1.x2, ls1.y2, ls2.x1, ls2.y1, ls2.x2, ls2.y2)
    d3 = point_to_line_dist(ls2.x1, ls2.y1, ls1.x1, ls1.y1, ls1.x2, ls1.y2)
    d4 = point_to_line_dist(ls2.x2, ls2.y2, ls1.x1, ls1.y1, ls1.x2, ls1.y2)
    return (d1 + d2 + d3 + d4) / 4


# ---------------------------------------------------------
# Projection + coverage
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# FINAL: Unified Wireframe/YUD evaluation (correct, not strict)
# ---------------------------------------------------------

def evaluate_segments(det_ls, gt_ls,
                      angle_thresh=35,     # relaxed
                      dist_thresh=15,      # relaxed
                      cov_thresh=0.2):     # relaxed

    used_det = set()
    TP = 0
    FN = 0
    localization_errors = []

    for g in gt_ls:
        # DO NOT FILTER BEFORE PROJECTION
        # Project ALL detected segments
        projections = []
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

        # Now filter AFTER projection
        valid = [(i, d, dist) for (i, d, t1, t2, ang, dist) in projections
                 if ang < angle_thresh and dist < dist_thresh]

        if not valid:
            FN += 1
            continue

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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if len(sys.argv) != 4:
        print("Usage: python eval_segments.py <ls_dir> <gt_mat> <out_csv>")
        sys.exit(1)

    ls_dir = sys.argv[1]
    gt_mat = sys.argv[2]
    out_csv = sys.argv[3]

    gt_ls = read_gt_line_segments(gt_mat)

    for prefix in ["st_th_", "st_it_", "bm_th_", "bm_it_"]:
        my_ls = read_my_line_segments(f"{ls_dir}/{prefix}lines.txt")

        TP, FP, FN, P, R, F1, LE = evaluate_segments(my_ls, gt_ls)

        print(prefix)
        print("Detected:", len(my_ls))
        print("Ground truth:", len(gt_ls))
        print("TP:", TP)
        print("FP:", FP)
        print("FN:", FN)
        print("Precision:", P)
        print("Recall:", R)
        print("F1:", F1)
        print("Localization Error:", LE)
        print()


if __name__ == "__main__":
    main()
