#ifndef LINES_HPP
#define LINES_HPP

#include <optional>

#include <opencv2/opencv.hpp>

#include "cand_type.hpp"
#include "cgraph_type.hpp"
#include "line_type.hpp"
#include "config.hpp"


__host__
std::vector<Line> extractLinesFromClusters(
    const std::vector<Cand>& candidates,
    const CandidateGraph& G,
    const std::vector<char>& edgeLabels
);

__host__
std::vector<std::vector<int>> retrieveClusters(
    const CandidateGraph& G,
    const std::vector<char>& edgeLabels
); // omits all 1-element clusters

__host__
std::optional<Line> clusterToLine(
    const std::vector<Cand>& candidates,
    const std::vector<int>& cluster
);

#endif