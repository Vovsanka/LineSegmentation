#ifndef COST_HPP
#define COST_HPP

#include <cmath>

#include <opencv2/opencv.hpp>

#include "cand_type.hpp"
#include "config.hpp"


__host__
double computeCandidateCost(
    const std::vector<Cand>& candidates,
    const Cand& cand1, 
    const Cand& cand2
);

__host__
double computeCandidateSimilarity( // [0, 1]
    const std::vector<Cand>& candidates,
    const Cand& cand1, 
    const Cand& cand2
);

__host__
bool checkNoGaps(
    const std::vector<Cand>& candidates,
    const Cand& cand1, 
    const Cand& cand2
);


#endif