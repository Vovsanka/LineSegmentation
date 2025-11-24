#ifndef BEST_HPP
#define BEST_HPP

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"
#include "show.hpp"

// Configuration start

// Configuration end


__global__
void bestScoreKernel(const uchar* F, double* S, int* D,
                     int width, int height);

__global__
void candidateThresholdKernel(const double *S, const uchar *D, uchar *C,
                              int width, int height);


#endif