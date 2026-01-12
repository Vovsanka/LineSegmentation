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
    return Cand(bestY, bestX, bestDir);
}

__host__
std::vector<Cand> sortThresholdCandidates(
    const double *S, size_t Sstep, 
    const int *D, size_t Dstep,
    int width, int height
) {
    std::vector<Cand> candList;
    int candCount = 0;
    std::vector<thrust::tuple<double,int>> scores;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double score = cell<double>(S, Sstep, y, x);
            double dir = cell<int>(D, Dstep, y, x);
            if (score >= CAND_THRESHOLD) {
                scores.push_back(thrust::make_tuple(-score, candCount++));
                candList.push_back(Cand(y, x, dir));
            }
        }
    }
    std::sort(std::begin(scores), std::end(scores));
    //
    std::vector<Cand> sortedCand(candCount);
    for (int k = 0; k < candCount; k++) {
        int ind = thrust::get<1>(scores[k]);
        sortedCand[k] = candList[ind];
    }
    return sortedCand;
}

__host__
bool isBlocked(const uchar* B, size_t Bstep, double y, double x, int width, int height) {
    int closestY = std::round(y);
    int closestX = std::round(x);
    if (closestY < 0 || height <= closestY) return true;
    if (closestX < 0 || width <= closestX) return true;
    return (cell<uchar>(B, Bstep, closestY, closestX)  != 0);
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
    const double *S, size_t Sstep,
    const int *D, size_t Dstep,
    int width, int height
) {
    cv::Mat BLOCKED(height, width, CV_8U, cv::Scalar(0)); 
    std::vector<Cand> chosenCand;
    //
    std::vector<Cand> tCand = sortThresholdCandidates(S, Sstep, D, Dstep, width, height);
    for (Cand startCand : tCand) {
        candidateExpand(F, Fstep, BLOCKED.ptr<uchar>(), BLOCKED.step, chosenCand, startCand, width, height);   
    }
    ///// debug start
    showMatrix(BLOCKED);
    std::cout << "#candidates = " << chosenCand.size() << std::endl;
    ///// debug end 
    return chosenCand;
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
    double score  = computeLabScore(F, Fstep, cand.y, cand.x, cand.dir, width, height);
    if (score < CAND_THRESHOLD) return;
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
    candidateExpand(
        F, Fstep, B, Bstep, chosenCand,
        Cand(cand.y + unitEdge1.y, cand.x + unitEdge1.x, cand.dir),
        width, height
    );
    candidateExpand(
        F, Fstep, B, Bstep, chosenCand,
        Cand(cand.y + unitEdge2.y, cand.x + unitEdge2.x, cand.dir),
        width, height
    );
}
