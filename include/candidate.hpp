#ifndef CANDIDATE_HPP
#define CANDIDATE_HPP

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "types.hpp"
#include "config.hpp"
#include "operations.hpp"
#include "score.hpp"



__global__ 
void bestPixelScoreKernel(
    const uchar* F, size_t Fstep,
    double* S, size_t Sstep,
    int* D, size_t Dstep,
    int width, int height
);

__host__
void computeBestPixelScores(
    cv::cuda::GpuMat& F,
    cv::cuda::GpuMat& S,
    cv::cuda::GpuMat& D
);

__host__ 
std::vector<Cand> extractSortedThresholdCandidates(
    cv::Mat& S, 
    cv::Mat& D
);





#endif