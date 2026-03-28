#ifndef LINES_HPP
#define LINES_HPP

#include <numeric>
#include <optional>

#include "cand_type.hpp"
#include "cgraph_type.hpp"
#include "line_type.hpp"
#include "config.hpp"


__host__
std::vector<Line> extractLinesFromClusters(
    const std::vector<Cand>& candidates,
    std::vector<std::vector<int>> clusters,
    int width, int height
);

__host__
std::optional<Line> clusterToLine(
    const std::vector<Cand>& candidates,
    const std::vector<int>& cluster,
    int width, int height
);

__host__
void computeLineImageIntersections( // Liang–Barsky line clipping
    double& tMin, double& tMax,
    double y0, double x0,
    double vy, double vx,
    int width, int height
);

#endif