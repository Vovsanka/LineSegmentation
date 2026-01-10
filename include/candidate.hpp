#ifndef CANDIDATE_HPP
#define CANDIDATE_HPP

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"
#include "operations.hpp"



__global__ 
void bestPixelScoreKernel(
    const uchar* F, size_t Fstep,
    double* S, size_t Sstep,
    int* D, size_t Dstep,
    int width, int height
);


__global__ 
void candidateThresholdKernel(
    const double *S, size_t Sstep,
    const int *D, size_t Dstep,
    uchar *C, size_t Cstep,
    int width, int height
);



#endif