#ifndef SCORE_HPP
#define SCORE_HPP

#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <cuda_runtime.h>
#include <thrust/pair.h>

#include "config.hpp"
#include "color.hpp"


__host__ __device__
thrust::pair<double,double> directionNormalUnitVector(int d);

__host__ __device__
double computeScore(const uchar* F,
                    double yPixel, double xPixel,
                    double unitNormY, double unitNormX,
                    int width, int height);

#endif