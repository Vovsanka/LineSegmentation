#ifndef SCORE_HPP
#define SCORE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "color.hpp"

__host__ __device__
float getPi();

__host__ __device__
thrust::tuple<float,float> getUnitVector(float rad);

__host__ __device__
float getRad(int direction);

__host__ __device__
float computeLabScore(const uchar* F,
                       float yPixel, float xPixel,
                       float dirRad, 
                       int width, int height); 

__host__ /*__device__*/
thrust::tuple<float,float> bestPossibleScore(const uchar* F,
                                               float yPixel, float xPixel,
                                               int width, int height);

#endif