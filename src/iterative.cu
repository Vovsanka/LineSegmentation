#include "iterative.hpp"


__host__ __device__
thrust::tuple<double,double,double> upgradeCandidate(const uchar* F,
                                                     thrust::tuple<double,double,double> cand,
                                                     int width, int height) {
    double yPixel = thrust::get<0>(cand);
    double xPixel = thrust::get<1>(cand);
    double dirRad = thrust::get<2>(cand);
    //
    thrust::tuple<double,double> unitNorm = getUnitVector(dirRad);
    double unitNormY = thrust::get<0>(unitNorm);
    double unitNormX = thrust::get<1>(unitNorm);
    //
    double bestScore = -1;
    double bestDir = 0;
    double bestY = 0, bestX = 0;
    for (int k = -UP_COUNT; k <= UP_COUNT; k++) {
            double y = yPixel + k*UP_STEP*unitNormY;
            double x = xPixel + k*UP_STEP*unitNormX;
            thrust::tuple<double,double> newScoreDir = bestPossibleScore(F, y, x, width, height);
            double newScore = thrust::get<0>(newScoreDir);
            double newDir = thrust::get<1>(newScoreDir);
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
std::vector<std::tuple<double,double,double>> sortThresholdCandidates(const double *S, int width, int height) {
    std::vector<std::tuple<double,double,double>> tCand;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            double score = S[idx];
            if (score >= START_THRESHOLD) {
                tCand.push_back(std::make_tuple(-score, y, x));
            }
        }
    }
    std::sort(std::begin(tCand), std::end(tCand));
    return tCand;
}

__host__
cv::Mat candidateIterativeSearch(const uchar* F, const double *S, const int *D, int width, int height) {
    std::vector<std::tuple<double,double,double>> tCand = sortThresholdCandidates(S, width, height);
    cv::Mat CI(height, width, CV_8U, cv::Scalar(0));
    for (std::tuple<double,double,double> start: tCand) {
        double startY = std::get<1>(start);
        double startX = std::get<2>(start);
        int idx = startY * width + startX;
        double direction = getRad(D[idx]);
        thrust::tuple<double,double,double> cand = thrust::make_tuple(startY, startX, direction);
        for (int k = 0; k < UP_ITERATIONS; k++) {
            cand = upgradeCandidate(F, cand, width, height);
        }
        //
        startY = thrust::get<0>(cand);
        startX = thrust::get<1>(cand);
        direction = thrust::get<2>(cand);
        candidateExpand(F, CI.ptr(), startY, startX, direction, width, height); 
        //
        showMatrix(CI);
    }
    return CI;
}


__host__
void candidateExpand(
    const uchar *F, uchar *CI, 
    double startY, double startX,
    double dirRad, 
    int width, int height) {
    //
    thrust::tuple<double,double> unitEdge = getUnitVector(dirRad + getPi()/2.0);
    double unitEdgeY = thrust::get<0>(unitEdge);
    double unitEdgeX = thrust::get<1>(unitEdge);
    //
    for (int k = 0; ; k++) {
        double y1 = startY + k*unitEdgeY;
        double x1 = startX + k*unitEdgeX;
        double y2 = startY - k*unitEdgeY;
        double x2 = startX - k*unitEdgeX;
        if (
            !setCandidates(F, CI, y1, x1, dirRad, width, height)
            &&
            !setCandidates(F, CI, y2, x2, dirRad, width, height)
        ) break;
    }
    return;
}


__host__ 
bool setCandidates(
    const uchar *F, uchar *CI, 
    double y, double x,
    double dirRad, 
    int width, int height) {
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
        double score = computeLabScore(F, downY, downX, dirRad, width, height);
        if (score >= MIN_THRESHOLD) {
            int idx = downY*width + downX;
            CI[idx] = 1;
            isSet = true;
        }
    }
    //
    if (downY >= 0 && upX < width) {
        double score = computeLabScore(F, downY, upX, dirRad, width, height);
        if (score >= THRESHOLD) {
            int idx = downY*width + upX;
            CI[idx] = 1;
            isSet = true;
        }
    }
    //
    if (upY < height && downX >= 0) {
        double score = computeLabScore(F, upY, downX, dirRad, width, height);
        if (score >= MIN_THRESHOLD) {
            int idx = upY*width + downX;
            CI[idx] = 1;
            isSet = true;
        }
    }
    //
    if (upY < height && upX < width) {
        double score = computeLabScore(F, upY, upX, dirRad, width, height);
        if (score >= MIN_THRESHOLD) {
            int idx = upY*width + upX;
            CI[idx] = 1;
            isSet = true;
        }
    }
    //
    if (isSet) {
        std::cout << "Set candidates for: y=" << y << " x=" << x  << std::endl;
    }
    return isSet;
}