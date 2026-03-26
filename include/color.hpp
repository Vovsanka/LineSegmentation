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

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getDirColorRgb(int dir); // dir in [0, 2*DIRECTIONS)

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getDirColorLab(int dir); // dir in [0, 2*DIRECTIONS)




#endif