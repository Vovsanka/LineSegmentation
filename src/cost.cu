#include "cost.hpp"


__host__
double computeCandidateCost(
    const uchar* F, size_t Fstep,
    std::vector<Cand> candidates,
    Cand& cand1, Cand& cand2
) { 
    // TODO 1: 0-cost if the distance is over 20px;
    // TODO 2: reward if really close < 2px (reward = 1)
    // TODO 3: penalty if directions do not match (penalty = 1) (almost allowed)
    // TODO 4: penalty if the cands are not on the same line (almost allowed) (line = pixel + dir) (penalty = 1)
    // TODO 4: reward if the line is continuous (no gaps, gap = 5px+) (reward = 1)
    // TODO 5: penalty if the line has gaps (penalty = 1)
}