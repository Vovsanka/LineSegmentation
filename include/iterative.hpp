#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"


__host__ __device__
thrust::tuple<double,double,double> upgradeCandidate(const uchar* F,
                                                     double yPixel, double xPixel,
                                                     double dirRad, 
                                                     int width, int height);

__host__
std::vector<std::tuple<double,double,double>> sortThresholdCandidates(const double *S, int width, int height);


__host__
void candidateIterativeSearch(const uchar* F, const double *S) {
    std::vector<std::tuple<double,double,double>> tCand = sortThresholdCandidates(S);
    for (std::tuple<double,double,double> start: tCand) {
        double startY = std::get<1>(start);
        double startX = std::get<2>(start);
    }
}



#endif