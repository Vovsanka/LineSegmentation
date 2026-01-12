#include "iterative.hpp"


__host__ __device__
Cand upgradeCandidate(
    const uchar* F, size_t Fstep,
    Cand cand,
    int width, int height
) {
    //
    Vec unitNorm = getUnitVector(cand.dir);
    //
    double bestScore = -1;
    int bestDir = 0;
    double bestY = 0, bestX = 0;
    for (int k = -UP_COUNT; k <= UP_COUNT; k++) {
            double y = cand.y + k*UP_STEP*unitNorm.y;
            double x = cand.x + k*UP_STEP*unitNorm.x;
            //
            thrust::tuple<double,int> newScoreDir = bestPossibleScore(F, Fstep, y, x, width, height);
            double newScore = thrust::get<0>(newScoreDir);
            int newDir = thrust::get<1>(newScoreDir);
            //
            if (newScore > bestScore) {
                bestScore = newScore;
                bestDir =  newDir;
                bestY = y;
                bestX = x;
            }
    }
    //
    return Cand(bestY, bestX, bestDir, bestScore);
}


__host__
bool isBlocked(const uchar* B, size_t Bstep, double y, double x, int width, int height) {
    int closestY = std::round(y);
    int closestX = std::round(x);
    if (closestY < 0 || height <= closestY) return true;
    if (closestX < 0 || width <= closestX) return true;
    return (cell<uchar>(B, Bstep, closestY, closestX) != 0);
}

__host__
void setBlocked(uchar*B, size_t Bstep, double y, double x, int width, int height) {
    int closestY = std::round(y);
    int closestX = std::round(x);
    if (closestY < 0 || height <= closestY) return;
    if (closestX < 0 || width <= closestX) return;
    cell<uchar>(B, Bstep, closestY, closestX) = 1;
}

__host__
std::vector<Cand> candidateIterativeSearch(
    const uchar* F, size_t Fstep,
    const std::vector<Cand>& tCandidates,
    int width, int height
) {
    cv::Mat BLOCKED(height, width, CV_8U, cv::Scalar(0)); 
    std::vector<Cand> chosenCandidates;
    //
    for (Cand startCand : tCandidates) {
        candidateExpand(
            F, Fstep,
            BLOCKED.ptr<uchar>(), BLOCKED.step,
            chosenCandidates,
            startCand,
            width, height
        );   
        //// debug start
        showMatrix(BLOCKED);
        //// debug end
    }
    //
    return chosenCandidates;
}

__host__ 
void candidateExpand(
    const uchar *F, size_t Fstep, 
    uchar* B, size_t Bstep,
    std::vector<Cand> &chosenCand,
    Cand cand,
    int width, int height
) {
    if (isBlocked(B, Bstep, cand.y, cand.x, width, height)) return;
    //
    for (int k = 0; k < UP_ITERATIONS; k++) {
        cand = upgradeCandidate(F, Fstep, cand, width, height);
    }
    if (isBlocked(B, Bstep, cand.y, cand.x, width, height)) return;
    if (cand.score < CAND_THRESHOLD) return;
    // 
    chosenCand.push_back(cand);
    setBlocked(B, Bstep, cand.y, cand.x, width, height);
    //
    int edge1 = getOrthogonalDirection(cand.dir);
    int edge2 = getOppositeDirection(edge1);
    //
    Vec unitEdge1 = getUnitVector(edge1);
    Vec unitEdge2 = getUnitVector(edge2);
    //
    int y1 = cand.y + unitEdge1.y;
    int x1 = cand.x + unitEdge1.x;
    double score1 = computeLabScore(F, Fstep, y1, x1, cand.dir, width, height);
    candidateExpand(
        F, Fstep, B, Bstep, chosenCand,
        Cand(y1, x1, cand.dir, score1),
        width, height
    );
    //
    int y2 = cand.y + unitEdge2.y;
    int x2 = cand.x + unitEdge2.x;
    double score2 = computeLabScore(F, Fstep, y2, x2, cand.dir, width, height);
    candidateExpand(
        F, Fstep, B, Bstep, chosenCand,
        Cand(y2, x2, cand.dir, score2),
        width, height
    );
}
