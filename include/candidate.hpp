#ifndef CANDIDATE_HPP
#define CANDIDATE_HPP

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"



__global__ 
void bestScoreKernel(
    const uchar* F, size_t Fstep,
    float* S, size_t Sstep,
    int* D, size_t Dstep,
    int width, int height
);



#endif