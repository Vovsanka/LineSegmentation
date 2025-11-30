#ifndef SCORE_HPP
#define SCORE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "color.hpp"

__host__ __device__
double getPi();

__host__ __device__
thrust::tuple<double,double> getUnitVector(double rad);

__host__ __device__
double getRad(int direction);

__host__ __device__
double computeLabScore(const uchar* F,
                       double yPixel, double xPixel,
                       double dirRad, 
                       int width, int height); 

__host__ /*__device__*/
thrust::tuple<double,double> bestPossibleScore(const uchar* F,
                                               double yPixel, double xPixel,
                                               int width, int height);

#endif