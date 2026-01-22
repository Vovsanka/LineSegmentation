#ifndef COST_HPP
#define COST_HPP

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
bool checkGaps(
    const std::vector<Cand>& candidates,
    const Cand& cand1, 
    const Cand& cand2
);


#endif