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
void bestScoreKernel(const uchar* F, float* S, int* D,
                     int width, int height);

__global__
void candidateThresholdKernel(const float *S, const int *D, uchar *C,
                              int width, int height);


#endif