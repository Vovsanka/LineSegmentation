#ifndef COST_HPP
#define COST_HPP


#include "config.hpp"
#include "cand_type.hpp"


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
double computeDirectionAngle(Vec unitNorm1, Vec unitNorm2);


__host__
bool checkNoGaps(
    const std::vector<Cand>& candidates,
    const Cand& cand1, 
    const Cand& cand2
);


#endif