#ifndef COLOR_HPP
#define COLOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>


const double TOL = 1e-6;


__host__ __device__
thrust::tuple<uchar,uchar,uchar> bicubicInterpolation(const uchar* F,
                                                      double y, double x,
                                                      int width, int height);

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getColorChannels(const uchar* F,
                                                  double y, double x,
                                                  int width, int height);

#endif