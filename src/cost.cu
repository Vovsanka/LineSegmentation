#include "cost.hpp"


__host__
double computeCandidateCost(
    const std::vector<Cand>& candidates,
    const Cand& cand1,
    const Cand& cand2
) {
    double sim = computeCandidateSimilarity(candidates, cand1, cand2);
    // logit of the similiarity
    if (sim <= TOL) return -COST_BOUND;
    if (sim >= 1.0 - TOL) return +COST_BOUND;
    double cost = std::log2(sim/(1.0 - sim));
    return max(min(cost, +COST_BOUND), -COST_BOUND);
}

__host__
double computeCandidateSimilarity( // [0, 1]
    const std::vector<Cand>& candidates,
    const Cand& cand1,
    const Cand& cand2
) {    
    // too close candidates => unclear similarity 
    double candDist = Cand::dist(cand1, cand2);
    if (candDist <= TOO_SMALL_DIST) {
        return 0.5;
    }
    // normal unit vector of the line segment between the candidates (in [0, PI])
    Vec orientSegment = Vec(cand2.y, cand2.x).subtract(Vec(cand1.y, cand1.x));
    Vec normalSegment(-orientSegment.x, orientSegment.y);
    Vec unitNormalSegment = normalSegment*(1.0/normalSegment.len());
    if (unitNormalSegment.y < 0) {
        unitNormalSegment = unitNormalSegment*(-1.0);
    } 
    // unit normal vectors of the candidate directions
    Vec unitNormal1 = getUnitVector(cand1.dir);
    Vec unitNormal2 = getUnitVector(cand2.dir);
    // angles between the candidate directions and the segment direction
    double angle12 = computeDirectionAngle(unitNormal1, unitNormal2);
    double angle1S = computeDirectionAngle(unitNormal1, unitNormalSegment);
    double angle2S = computeDirectionAngle(unitNormal2, unitNormalSegment);
    // handle all cases 
    if (angle12 <= SIMILAR_DIR_ANGLE) { 
        // similar candidate directions
        if (angle1S <= SIMILAR_DIR_ANGLE && angle2S <= SIMILAR_DIR_ANGLE) {
            // candidate directions similar to the segment direction 
            // => same line (same line segment if in proximity) => similar
            double minDistToLine = min(cand1.distToLine(cand2), cand2.distToLine(cand1));
            double goodSim = max(0.5, 1.0 - minDistToLine/(GOOD_DIST_FACTOR*candDist));
            return goodSim; 
        }
        // candidate directions differ from the segment direction
        if (isEmptySpace(candidates, cand1, cand2)) {
            // parallel lines => dissimilar
            double minDistToLine = min(cand1.distToLine(cand2), cand2.distToLine(cand1));
            double badSim = max(0.0, min(0.5, 1.0 - minDistToLine/(BAD_DIST_FACTOR*candDist)));
            return badSim; 
        }
        // no gaps => unclear similarity
        return 0.5;
    }
    // dissimilar candidate directions
    if (angle1S > SIMILAR_DIR_ANGLE && angle2S > SIMILAR_DIR_ANGLE && isEmptySpace(candidates, cand1, cand2)) {
        // candidates represent different line segments => dissimilar
        double minDistToLine = min(cand1.distToLine(cand2), cand2.distToLine(cand1));
        double badSim = max(0.0, min(0.5, 1.0 - minDistToLine/(BAD_DIST_FACTOR*candDist)));
        return badSim;
    }
    // at least candidate direction similar to the segment directions => unclear similarity
    // dissimilar directions but no gaps => unclear similarity
    return 0.5;
}

__host__
double computeDirectionAngle(Vec unitNorm1, Vec unitNorm2) { 
    double angle = acos(min(1.0, max(-1.0, unitNorm1.dot(unitNorm2))));
    return min(angle, PI - angle); // angles between the lines are up to PI/2
}

__host__
bool isEmptySpace(
    const std::vector<Cand>& candidates,
    const Cand& cand1, 
    const Cand& cand2
) { 
    double candDist = Cand::dist(cand1, cand2);
    //
    double maxTriangleDist = LINE_TRIANGLE_FACTOR*candDist;
    for (int k = 0; k < candidates.size(); k++) {
        double side1 = Cand::dist(cand1, candidates[k]);
        if (side1 > candDist || side1 < TOL) continue;
        double side2 =  Cand::dist(cand2, candidates[k]); 
        if (side2 > candDist || side2 < TOL) continue;
        if (side1 + side2 <= maxTriangleDist) {
            return false;
        }
    }
    //
    return true;
}