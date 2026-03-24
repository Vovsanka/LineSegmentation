#include "cost.hpp"


double computeCandidateCost(
    const std::vector<Cand>& candidates,
    const Cand& cand1,
    const Cand& cand2
) {
    double sim = computeCandidateSimilarity(candidates, cand1, cand2);
    // logit of the similiarity
    if (sim  < TOL) return -COST_BOUND;
    if (sim >= 1.0 - TOL) return +COST_BOUND;
    double cost = std::log2(sim/(1.0 - sim));
    return max(min(cost, +COST_BOUND), -COST_BOUND);
}

double computeCandidateSimilarity( // [0, 1]
    const std::vector<Cand>& candidates,
    const Cand& cand1,
    const Cand& cand2
) { 
    //
    int dirDiff = getDirDifference(cand1.dir, cand2.dir);
    double dirSim = 1.0 - (2.0*dirDiff)/DIRECTIONS;
    //
    double distSim1 = 1.0 - cand1.distToLine(cand2)/(2*GOOD_DIST_TO_CAND_LINE);
    double distSim2 = 1.0 - cand2.distToLine(cand1)/(2*GOOD_DIST_TO_CAND_LINE);
    double distSim = max(0.0, max(distSim1, distSim2));
    //
    return dirSim*distSim;
    // // reward if the line is continuous (no gaps)
    // if (checkNoGaps(candidates, cand1, cand2)) {
    //     return sim;
    // } 
    // // line has gaps => dissimilar
    // return 0.0;
}

// bool checkNoGaps(
//     const std::vector<Cand>& candidates,
//     const Cand& cand1, 
//     const Cand& cand2
// ) { 
//     double candDist = Cand::dist(cand1, cand2);
//     if (candDist < MIN_GAP_SIZE) return true;
//     //
//     double maxTriangleDist = LINE_TRIANGLE_FACTOR*candDist;
//     std::vector<int> segmentCandidates;
//     for (int k = 0; k < candidates.size(); k++) {
//         double side1 = Cand::dist(cand1, candidates[k]);
//         double side2 =  Cand::dist(cand2, candidates[k]); 
//         if (
//             side1 <= candDist + TOL &&
//             side2 <= candDist + TOL && 
//             side1 + side2 <= maxTriangleDist
//         ) {
//             segmentCandidates.push_back(k);
//         }
//     }
//     //
//     Vec lineVec(cand2.y - cand1.y, cand2.x - cand1.x);
//     std::vector<double> projections;
//     for (int k : segmentCandidates) {
//         const Cand& cand = candidates[k];
//         Vec v1(cand.y - cand1.y, cand.x - cand1.x);
//         double t = v1.dot(lineVec)/lineVec.dot(lineVec);
//         double projectionDist = t*lineVec.len(); // dist from cand1 
//         projections.push_back(projectionDist);
//     }
//     // check for the gaps in the projections
//     std::sort(std::begin(projections), std::end(projections));
//     int lastProj = 0;
//     for (double p : projections) {
//         if (p + TOL < 0  || p - TOL > candDist) continue;
//         if (p - lastProj > MIN_GAP_SIZE) return false;
//         lastProj = p;
//     }
//     return true;
// }