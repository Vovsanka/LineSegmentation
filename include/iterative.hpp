#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"
#include "operations.hpp"


__host__ __device__
thrust::tuple<float,float,int> upgradeCandidate(
    const uchar* F, size_t Fstep,
    thrust::tuple<float,float,int> cand,
    int width, int height
);

__host__
std::vector<std::tuple<float,int,int>> sortThresholdCandidates(
    const float *S, size_t Sstep, 
    int width, int height
);

__host__
cv::Mat candidateIterativeSearch(
    const uchar* F, size_t Fstep,
    const float *S, size_t Sstep,
    const int *D, size_t Dstep,
    int width, int height
);


__host__
void candidateExpand(
    const uchar *F, size_t Fstep, 
    uchar *CI, size_t CIstep, 
    float startY, float startX,
    int dir, 
    int width, int height
);


__host__ 
bool setCandidates(
    const uchar *F, size_t Fstep,
    uchar *CI, size_t CIstep, 
    float y, float x,
    int dir, 
    int width, int height
);

#endif