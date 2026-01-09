#include "iterative.hpp"


__host__ __device__
Cand make_candidate(double y, double x, int dir) {
    return thrust::make_tuple(y, x, dir);
}

__host__ __device__
Cand upgradeCandidate(
    const uchar* F, size_t Fstep,
    Cand cand,
    int width, int height
) {
    double yPixel = thrust::get<0>(cand);
    double xPixel = thrust::get<1>(cand);
    int dir = thrust::get<2>(cand);
    //
    thrust::tuple<double,double> unitNorm = getUnitVector(dir);
    double unitNormY = thrust::get<0>(unitNorm);
    double unitNormX = thrust::get<1>(unitNorm);
    //
    double bestScore = -1;
    int bestDir = 0;
    double bestY = 0, bestX = 0;
    for (int k = -UP_COUNT; k <= UP_COUNT; k++) {
            double y = yPixel + k*UP_STEP*unitNormY;
            double x = xPixel + k*UP_STEP*unitNormX;
            thrust::tuple<double,int> newScoreDir = bestPossibleScore(F, Fstep, y, x, width, height);
            double newScore = thrust::get<0>(newScoreDir);
            int newDir = thrust::get<1>(newScoreDir);
            if (newScore > bestScore) {
                bestScore = newScore;
                bestDir =  newDir;
                bestY = y;
                bestX = x;
            }
    }
    //
    return thrust::make_tuple(bestY, bestX, bestDir);
}

__host__
std::vector<Cand> sortThresholdCandidates(
    const double *S, size_t Sstep, 
    const int *D, size_t Dstep,
    int width, int height
) {
    std::vector<thrust::tuple<double,Cand>> tCand;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double score = cell<double>(S, Sstep, y, x);
            double dir = cell<int>(D, Dstep, y, x);
            if (score >= CAND_THRESHOLD) {
                tCand.push_back(thrust::make_tuple(-score, make_candidate(y, x, dir)));
            }
        }
    }
    std::sort(std::begin(tCand), std::end(tCand));
    //
    std::vector<Cand> sortedCand(tCand.size());
    int ind = 0;
    for (auto& [minusScore, cand] : tCand) {
        sortedCand[ind++] = cand;
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
    for (Cand start : tCand) {
        int startY = thrust::get<1>(start);
        int startX = thrust::get<2>(start);
        if (isBlocked(BLOCKED.ptr<uchar>(), BLOCKED.step, startY, startX, width, height)) continue;
        int startDir = cell<int>(D, Dstep, startY, startX);
        Cand cand = thrust::make_tuple(startY, startX, startDir);
        for (int k = 0; k < UP_ITERATIONS; k++) {
            cand = upgradeCandidate(F, Fstep, cand, width, height);
        }
        startY = thrust::get<0>(cand);
        startX = thrust::get<1>(cand);
        if (isBlocked(BLOCKED.ptr<uchar>(), BLOCKED.step, startY, startX, width, height)) continue;
        //
        candidateExpand(F, Fstep, BLOCKED.ptr<uchar>(), BLOCKED.step, chosenCand, cand, width, height);   
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
    double startY = thrust::get<0>(cand);
    double startX = thrust::get<1>(cand);
    int dir = thrust::get<2>(cand);
    //
    int edge1 = getOrthogonalDirection(dir);
    int edge2 = getOppositeDirection(edge1);
    //
    thrust::tuple<double,double> unitEdge1 = getUnitVector(edge1);
    double unitEdge1Y = thrust::get<0>(unitEdge1);
    double unitEdge1X = thrust::get<1>(unitEdge1);
    //
    thrust::tuple<double,double> unitEdge2 = getUnitVector(edge2);
    double unitEdge2Y = thrust::get<0>(unitEdge2);
    double unitEdge2X = thrust::get<1>(unitEdge2);
    //
    for (double y = startY, x = startX; !isBlocked(B, Bstep, y, x, width, height); y += unitEdge1Y, x += unitEdge1X) {
        double score  = computeLabScore(F, Fstep, y, x, dir, width, height);
        if (score < CAND_THRESHOLD) break;
        chosenCand.push_back(make_candidate(y, x, dir));
        setBlocked(B, Bstep, y, x, width, height);
    }
    //
    for (double y = startY + unitEdge2Y, x = startX + unitEdge2X; !isBlocked(B, Bstep, y, x, width, height); y += unitEdge2Y, x += unitEdge2X) {
        double score  = computeLabScore(F, Fstep, y, x, dir, width, height);
        if (score < CAND_THRESHOLD) break;
        chosenCand.push_back(make_candidate(y, x, dir));
        setBlocked(B, Bstep, y, x, width, height);
    }
}
