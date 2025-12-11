#ifndef BEST_HPP
#define BEST_HPP

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"

// Configuration start

// Configuration end


__global__
void bestScoreKernel(const uchar* F, float* S, int* D,
                     int width, int height);

__global__
void candidateThresholdKernel(const float *S, const int *D, uchar *C,
                              int width, int height);


#endif