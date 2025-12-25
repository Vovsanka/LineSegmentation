#include "iterative.hpp"


__host__ __device__
thrust::tuple<double,double,int> upgradeCandidate(
    const uchar* F, size_t Fstep,
    thrust::tuple<double,double,int> cand,
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
std::vector<std::tuple<double,int,int>> sortThresholdCandidates(
    const double *S, size_t Sstep, 
    int width, int height
) {
    std::vector<std::tuple<double,int,int>> tCand;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double* rowS = (double*)((uchar*)S + y * Sstep);
            double score = rowS[x];
            if (score >= HIGH_THRESHOLD) {
                tCand.push_back(std::make_tuple(-score, y, x));
            }
        }
    }
    std::sort(std::begin(tCand), std::end(tCand));
    return tCand;
}

__host__
cv::Mat candidateIterativeSearch(
    const uchar* F, size_t Fstep,
    const double *S, size_t Sstep,
    const int *D, size_t Dstep,
    int width, int height
) {
    std::vector<std::tuple<double,int,int>> tCand = sortThresholdCandidates(S, Sstep, width, height);
    cv::Mat CI(height, width, CV_8U, cv::Scalar(0)); // TODO: fractional candidates instead of CI
    for (std::tuple<double,int,int> start : tCand) {
        int startY = std::get<1>(start);
        int startX = std::get<2>(start);
        int* rowD = (int*)((uchar*)D + startY * Dstep);
        int startDir = rowD[startX];
        thrust::tuple<double,double,int> cand = thrust::make_tuple(startY, startX, startDir);
        for (int k = 0; k < UP_ITERATIONS; k++) {
            cand = upgradeCandidate(F, Fstep, cand, width, height);
        }
        //
        double candY = thrust::get<0>(cand);
        double candX = thrust::get<1>(cand);
        int candDir = thrust::get<2>(cand);
        candidateExpand(F, Fstep, CI.ptr<uchar>(), CI.step, candY, candX, candDir, width, height); 
        //
        showMatrix(CI);
    }
    return CI;
}


__host__
void candidateExpand(
    const uchar *F, size_t Fstep, 
    uchar *CI, size_t CIstep, 
    double startY, double startX,
    int dir, 
    int width, int height
) {
    //
    thrust::tuple<double,double> unitEdge = getOrthogonalUnitVector(dir);
    double unitEdgeY = thrust::get<0>(unitEdge);
    double unitEdgeX = thrust::get<1>(unitEdge);
    //
    for (int k = 0; ; k++) {
        double y1 = startY + k*ITER_STEP*unitEdgeY; 
        double x1 = startX + k*ITER_STEP*unitEdgeX;
        if (!setCandidates(F, Fstep, CI, CIstep, y1, x1, dir, width, height)) break;
    }
    //
    for (int k = 0; ; k++) {
        double y2 = startY - k*ITER_STEP*unitEdgeY;
        double x2 = startX - k*ITER_STEP*unitEdgeX;
        if (!setCandidates(F, Fstep, CI, CIstep, y2, x2, dir, width, height)) break;
    }
    return;
}


__host__ 
bool setCandidates(
    const uchar *F, size_t Fstep,
    uchar *CI, size_t CIstep, 
    double y, double x,
    int dir, 
    int width, int height
) {
    //
    if (
        y < 0 || y >= height ||
        x < 0 || x >= width
    ) return false;
    //
    int downY = floor(y);
    int downX = floor(x);
    int upY = ceil(y);
    int upX = ceil(x);
    //
    bool isSet = false;
    //
    if (downY >= 0 && downX >= 0) {
        double score = computeLabScore(F, Fstep, downY, downX, dir, width, height);
        if (score >= LOW_THRESHOLD) {
            int* rowCI = (int*)((uchar*)CI + downY * CIstep);
            rowCI[downX] = 1;
            isSet = true;
        }
    }
    //
    if (downY >= 0 && upX < width) {
        double score = computeLabScore(F, Fstep, downY, upX, dir, width, height);
        if (score >= LOW_THRESHOLD) {
            int* rowCI = (int*)((uchar*)CI + downY * CIstep);
            rowCI[upX] = 1;
            isSet = true;
        }
    }
    //
    if (upY < height && downX >= 0) {
        double score = computeLabScore(F, Fstep, upY, downX, dir, width, height);
        if (score >= LOW_THRESHOLD) {
            int* rowCI = (int*)((uchar*)CI + upY * CIstep);
            rowCI[downX] = 1;
            isSet = true;
        }
    }
    //
    if (upY < height && upX < width) {
        double score = computeLabScore(F, Fstep, upY, upX, dir, width, height);
        if (score >= LOW_THRESHOLD) {
            int* rowCI = (int*)((uchar*)CI + upY * CIstep);
            rowCI[upX] = 1;
            isSet = true;
        }
    }
    //
    if (isSet) {
        std::cout << "Set candidates for: y=" << y << " x=" << x  << std::endl;
    }
    return isSet;
}