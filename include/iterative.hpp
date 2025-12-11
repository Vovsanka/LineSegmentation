#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"


__host__ __device__
thrust::tuple<float,float,float> upgradeCandidate(const uchar* F,
                                                     thrust::tuple<float,float,float>,
                                                     int width, int height);

__host__
std::vector<std::tuple<float,float,float>> sortThresholdCandidates(const float *S, int width, int height);


__host__
cv::Mat candidateIterativeSearch(const uchar* F, const float *S, const int *D, int width, int height);


__host__
void candidateExpand(
    const uchar *F, uchar *CI, 
    float startY, float startX,
    float dirRad, 
    int width, int height);


__host__ 
bool setCandidates(
    const uchar *F, uchar *CI, 
    float y, float x,
    float dirRad, 
    int width, int height);

#endif