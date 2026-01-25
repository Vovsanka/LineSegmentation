#include "lines.hpp"

__host__
std::vector<Line> extractLinesFromClusters(
    const std::vector<Cand>& candidates,
    const CandidateGraph& G,
    const std::vector<char>& edgeLabels
) {
    std::vector<std::vector<int>> clusters = retrieveClusters(G, edgeLabels);
    //
    std::vector<Line> lines;
    for (const std::vector<int>& cluster : clusters) {
        std::optional<Line> line = clusterToLine(candidates, cluster);
        if (line.has_value()) {
            lines.push_back(*line);
        }
    }
    return lines;
}

__host__
std::vector<std::vector<int>> retrieveClusters( 
    const CandidateGraph& G,
    const std::vector<char>& edgeLabels
) {
    std::vector<int> clusterMap(G.n, -1);
    int clusterCount = 0;
    for (int k = 0; k < G.edges.size(); k++) {
        if (edgeLabels[k] == 0) {
            const Edge& e = G.edges[k]; 
            if (clusterMap[e.c1] == -1 && clusterMap[e.c2] == -1) {
                clusterMap[e.c1] = clusterMap[e.c2] = clusterCount++; 
            } else if (clusterMap[e.c1] == -1) { // clusterMap[e.c2] != -1
                clusterMap[e.c1] = clusterMap[e.c2];
            } else if (clusterMap[e.c2] == -1) { // clusterMap[e.c1] != -1
                clusterMap[e.c2] = clusterMap[e.c1];
            }
        }
    }
    // gather all the clusters with 2+ nodes
    std::vector<std::vector<int>> clusters(clusterCount, std::vector<int>());
    for (int node = 0; node < G.n; node++) {
        if (clusterMap[node] != -1) {
            clusters[clusterMap[node]].push_back(node);
        }
    }
    return clusters;
}

__host__
std::optional<Line> clusterToLine (
    const std::vector<Cand>& candidates,
    const std::vector<int>& cluster
) {
    if (cluster.size() < MIN_LINE_SIZE) return std::nullopt;
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
    Vec lineVec(vy, vx);
    double minT = +1e6, maxT = -1e6; // projection factor
    for (int node : cluster) {
        const Cand& cand = candidates[node];
        Vec v1(cand.y - y0, cand.x - x0);
        double t = v1.dot(lineVec)/lineVec.dot(lineVec);
        minT = min(t, minT);
        maxT = max(t, maxT);
    }
    // 
    double end1Y = y0 + minT*vy;
    double end1X = x0 + minT*vx;
    double end2Y = y0 + maxT*vy;
    double end2X = x0 + maxT*vx;
    return Line(end1Y, end1X, end2Y, end2X); // TODO: fix min-max-T
}
