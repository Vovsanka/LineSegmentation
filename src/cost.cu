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
    if (candDist <= LINE_THICKNESS) {
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
    double angleCandidates = computeDirectionAngle(unitNormal1, unitNormal2);
    double angle1 = computeDirectionAngle(unitNormal1, unitNormalSegment);
    double angle2 = computeDirectionAngle(unitNormal2, unitNormalSegment);
    // similarity based on candidate distance to the line of the other candidate
    double distToLine = min(cand1.distToLine(cand2), cand2.distToLine(cand1));
    double sim = max(0.0, 1.0 - distToLine/LINE_THICKNESS);
    // handle all cases 
    if (angleCandidates < DISSIMILAR_DIR_ANGLE) {
        // candidate directions are not dissimilar
        if (min(angle1, angle2) < DISSIMILAR_DIR_ANGLE) {
            // directions are not dissimilar
            if (angleCandidates <= SIMILAR_DIR_ANGLE) { 
                // similar candidate directions
                if (angle1 <= SIMILAR_DIR_ANGLE && angle2 <= SIMILAR_DIR_ANGLE) {
                    // candidate directions similar to the segment direction 
                    if (checkNoGaps(candidates, cand1, cand2)) {
                        // same line => similar
                        return max(0.5, sim); 
                    }
                    // gaps => unclear similarity
                    return 0.5;
                }
            }
            // directions are not similar
            if (checkNoGaps(candidates, cand1, cand2)) {
                // not dissimilar directions => a little similar candidates;
                return 0.6;
            }
        }
        // candidate directions are different from the segment direction
        if (isEmptySpace(candidates, cand1, cand2)) {
            // parallel lines => similarity is based on the distance (possibly unite)
            return sim; 
        }
    }
    // directions are dissimilar
    if (min(angle1, angle2) > SIMILAR_DIR_ANGLE && isEmptySpace(candidates, cand1, cand2)) {
        // candidates represent different line segments => dissimilar    
        return min(0.5, sim);
    }
    // at leastone  candidate direction similar to the segment directions => unclear similarity
    // dissimilar directions but no gaps => unclear similarity
    return 0.5;
}

__host__
double computeDirectionAngle(Vec unitNorm1, Vec unitNorm2) { 
    double angle = acos(min(1.0, max(-1.0, unitNorm1.dot(unitNorm2))));
    return min(angle, PI - angle); // angles between the lines are up to PI/2
}

__host__
bool checkNoGaps(
    const std::vector<Cand>& candidates,
    const Cand& cand1, 
    const Cand& cand2
) { 
    double candDist = Cand::dist(cand1, cand2);
    if (candDist < LINE_THICKNESS) return true;
    //
    double maxTriangleDist = LINE_TRIANGLE_FACTOR*candDist;
    std::vector<int> segmentCandidates;
    for (int k = 0; k < candidates.size(); k++) {
        double side1 = Cand::dist(cand1, candidates[k]);
        double side2 =  Cand::dist(cand2, candidates[k]); 
        if (
            side1 <= candDist + TOL &&
            side2 <= candDist + TOL && 
            side1 + side2 <= maxTriangleDist
        ) {
            segmentCandidates.push_back(k);
        }
    }
    //
    Vec lineVec(cand2.y - cand1.y, cand2.x - cand1.x);
    std::vector<double> projections;
    for (int k : segmentCandidates) {
        const Cand& cand = candidates[k];
        Vec v1(cand.y - cand1.y, cand.x - cand1.x);
        double t = v1.dot(lineVec)/lineVec.dot(lineVec);
        double projectionDist = t*lineVec.len(); // dist from cand1 
        projections.push_back(projectionDist);
    }
    // check for the gaps in the projections
    std::sort(std::begin(projections), std::end(projections));
    int lastProj = 0;
    for (double p : projections) {
        if (p + TOL < 0  || p - TOL > candDist) continue;
        if (p - lastProj > LINE_THICKNESS) return false;
        lastProj = p;
    }
    return true;
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
    return true; 
}
