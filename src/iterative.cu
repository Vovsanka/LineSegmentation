#include "iterative.hpp"


__host__ __device__
thrust::tuple<double,double,double> upgradeCandidate(const uchar* F,
                                                     double yPixel, double xPixel,
                                                     double score, double dirRad, 
                                                     int width, int height) {
    thrust::tuple<double,double> unitNorm = getUnitVector(dirRad);
    double unitNormY = thrust::get<0>(unitNorm);
    double unitNormX = thrust::get<1>(unitNorm);
    //
    double bestScore = -1;
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
                bestY = y;
                bestX = x;
            }
        }
    }
    //
    return thrust::make_tuple(bestScore, bestY, bestX);
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
