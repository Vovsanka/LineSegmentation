#ifndef COLOR_HPP
#define COLOR_HPP

#include "operations.hpp"


__host__ __device__
uchar bicubicInterpolation1(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int width, int height
);

__host__ __device__
thrust::tuple<uchar,uchar,uchar> bicubicInterpolation3(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int width, int height
);

__host__ __device__
uchar getGrayColor(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int width, int height
);

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getColorChannels(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int width, int height
);



#endif