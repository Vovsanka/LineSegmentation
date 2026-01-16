#ifndef COST_HPP
#define COST_HPP

#include <opencv2/opencv.hpp>

#include "cand_type.hpp"
#include "config.hpp"


__host__
double computeCandidateCost(
    std::vector<Cand> candidates,
    Cand& cand1, Cand& cand2
);

__host__
bool checkGaps(
    std::vector<Cand> candidates,
    Cand& cand1, Cand& cand2
);


#endif