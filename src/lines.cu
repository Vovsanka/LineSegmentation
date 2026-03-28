#include "lines.hpp"

__host__
std::vector<Line> extractLinesFromClusters(
    const std::vector<Cand>& candidates,
    std::vector<std::vector<int>> clusters,
    int width, int height
) {
    //
    std::vector<Line> lines;
    for (const std::vector<int>& cluster : clusters) {
        std::optional<Line> line = clusterToLine(candidates, cluster, width, height);
        if (line.has_value()) {
            lines.push_back(*line);
        }
    }
    return lines;
}

__host__
std::optional<Line> clusterToLine (
    const std::vector<Cand>& candidates,
    const std::vector<int>& cluster,
    int width, int height
) {
    if (cluster.size() < MIN_LINE_CLUSTER) return std::nullopt;
    //
    std::vector<cv::Point2d> points;
    for (int node : cluster) {
        const Cand& cand = candidates[node];
        points.push_back({cand.x, cand.y});
    }
    //
    cv::Vec4f line;
    cv::fitLine(points, line, cv::DIST_L2, 0, 0.01, 0.01);
    // line = (vx, vy, x0, y0)
    float vx = line[0];
    float vy = line[1];
    float x0 = line[2];
    float y0 = line[3];
    //
    Vec originVec(y0, x0);
    Vec lineVec(vy, vx);
    //
    double minT = +INF, maxT = -INF; // projection factors
    //
    for (int node : cluster) {
        const Cand& cand = candidates[node];
        Vec candVec(cand.y, cand.x);
        Vec v1 = candVec.subtract(originVec);
        double t = v1.dot(lineVec)/lineVec.dot(lineVec); // TODO: clamp projections to the image edges
        minT = min(t, minT);
        maxT = max(t, maxT);
    }
    //
    double imgMinT = -INF, imgMaxT = +INF;
    computeLineImageIntersections(imgMinT, imgMaxT, y0, x0, vy, vx, width, height);
    //
    minT = max(minT, imgMinT);
    maxT = min(maxT, imgMaxT);
    // 
    double end1Y = y0 + minT*vy;
    double end1X = x0 + minT*vx;
    double end2Y = y0 + maxT*vy;
    double end2X = x0 + maxT*vx;
    //
    return Line(end1Y, end1X, end2Y, end2X);
}

__host__
void computeLineImageIntersections(
    double& tMin, double& tMax,
    double y0, double x0,
    double vy, double vx,
    int width, int height
) {
    auto update = [&](double p, double q) { 
        if (p == 0) return q >= 0;
        double r = q / p;
        if (p < 0) {
            if (r > tMax) return false;
            if (r > tMin) tMin = r;
        } else {
            if (r < tMin) return false;
            if (r < tMax) tMax = r;
        }
        return true;
    };
    //
    if (!update(-vx, x0)) return;
    //
    if (!update( vx, (width - 1) - x0)) return;
    //
    if (!update(-vy, y0)) return;
    //
    if (!update( vy, (height - 1) - y0)) return;
}



