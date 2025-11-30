#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"


__host__ __device__
thrust::tuple<double,double,double> upgradeCandidate(const uchar* F,
                                                     thrust::tuple<double,double,double>,
                                                     int width, int height);

__host__
std::vector<std::tuple<double,double,double>> sortThresholdCandidates(const double *S, int width, int height);


__host__
cv::Mat candidateIterativeSearch(const uchar* F, const double *S, const int *D, int width, int height);


__host__
void candidateExpand(
    const uchar *F, uchar *CI, 
    double startY, double startX,
    double dirRad, 
    int width, int height);


__host__ 
bool setCandidates(
    const uchar *F, uchar *CI, 
    double y, double x,
    double dirRad, 
    int width, int height);

#endif