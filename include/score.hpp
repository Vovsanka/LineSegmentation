#ifndef SCORE_HPP
#define SCORE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <cuda_runtime.h>
#include <thrust/tuple.h>

#include "config.hpp"
#include "color.hpp"


__host__ __device__
thrust::tuple<double,double> directionNormalUnitVector(int d);

__host__ __device__
double computeScore(const uchar* F,
                    double yPixel, double xPixel,
                    int direction,
                    int width, int height);

#endif