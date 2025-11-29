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
    for (int i = -UP_COUNT; i <= UP_COUNT; i++) {
        for (int j = -UP_COUNT; j <= UP_COUNT; j++) {
            double y = yPixel + i*UP_STEP*unitNormY;
            double x = xPixel + i*UP_STEP*unitNormX;
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
            if (score >= THRESHOLD) {
                tCand.push_back(std::make_tuple(-score, y, x));
            }
        }
    }
    std::sort(std::begin(tCand), std::end(tCand));
    return tCand;
}

__host__
void candidateIterativeSearch(const uchar* F, const double *S, const double *D, int width, int height) {
    std::vector<std::tuple<double,double,double>> tCand = sortThresholdCandidates(S, width, height);
    for (std::tuple<double,double,double> start: tCand) {
        double startY = std::get<1>(start);
        double startX = std::get<2>(start);
        int idx = startY * width + startX;
        double direction = getRad(D[idx]);
        thrust::tuple<double,double,double> cand = thrust::make_tuple(startY, startX, direction);
        for (int k = 0; k < UP_ITERATIONS; k++) {
            cand = upgradeCandidate(F, cand, width, height);
        }
        
    }
}
