import sys

from scipy.io import loadmat


class LineSegment:

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


def read_my_line_segments(ls_txt: str) -> list[LineSegment]:
    my_ls: list[LineSegment] = []
    with open(ls_txt, "r") as f:
        n = int(f.readline().strip())
        for _ in range(n):
            line = f.readline().strip()
            x1, y1, x2, y2 = map(float, line.split())
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
    gt_ls: list[LineSegment] = []
    for x1, y1, x2, y2 in arr:
        gt_ls.append(LineSegment(float(x1), float(y1), float(x2), float(y2)))
    return gt_ls


def main():

    if len(sys.argv) != 4:
        sys.exit(1)

    ls_txt: str = sys.argv[1]
    gt_mat: str = sys.argv[2]
    out_csv: str = sys.argv[3]

    my_ls: list[LineSegment] = read_my_line_segments(ls_txt)
    gt_ls: list[LineSegment] = read_gt_line_segments(gt_mat)

    print(len(my_ls))
    print(len(gt_ls))
    



if __name__ == "__main__":
    main()
