#include "cost.hpp"


__host__
double computeCandidateCost(
    std::vector<Cand> candidates,
    Cand& cand1, Cand& cand2
) { 
    // reward if candidates are really close
    if (Cand::dist(cand1, cand2) <= ALMOST_SAME_PIXEL) return -1; 
    // penalty if directions do not match
    if (Cand::dirDiff(cand1, cand2) > ALMOST_SAME_DIR) return 1;
    // penalty if the cands are not on the same line
    if (
        cand1.distToLine(cand2) > ALMOST_SAME_LINE && 
        cand2.distToLine(cand1) > ALMOST_SAME_LINE
    ) return 1;
    // reward if the line is continuous (no gaps)
    if (checkGaps(candidates, cand1, cand2)) return -1;
    // penalty if the line has gaps
    return 1;
}

__host__
bool checkGaps(
    std::vector<Cand> candidates,
    Cand& cand1, Cand& cand2
) { 
    double candDist = Cand::dist(cand1, cand2);
    if (candDist < GAP_SIZE) return true;
    //
    double maxTriangleDist = candDist + ALMOST_LINE_TRIANGLE;
    std::vector<int> segmentCandidates;
    for (int k = 0; k < candidates.size(); k++) {
        double triangleDist = Cand::dist(cand1, candidates[k]) + Cand::dist(cand2, candidates[k]);
        if (triangleDist <= ALMOST_LINE_TRIANGLE) {
            segmentCandidates.push_back(k);
        }
    }
    //
    Vec lineVec(cand2.y - cand1.y, cand2.x - cand1.x);
    std::vector<double> projections;
    for (int k : segmentCandidates) {
        Cand& cand = candidates[k];
        Vec v1(cand.y - cand1.y, cand.x - cand1.x);
        double t = v1.dot(lineVec)/lineVec.dot(lineVec);
        double projectionDist = t*lineVec.len();
        projections.push_back(projectionDist);
    }
    // check for the gaps in the projections
    std::sort(std::begin(projections), std::end(projections));
    int lastProj = 0;
    for (double p : projections) {
        if (p <= 0 || p >= candDist) continue;
        if (p - lastProj > GAP_SIZE) return false;
        lastProj = p;
    }
    return true;
}