#ifndef DIRECTIONS_HPP
#define DIRECTIONS_HPP

#include "config.hpp"
#include "vec_type.hpp"


__host__ __device__
double getRad(int d); // d in [0, 2*DIRECTIONS), returns [0, 2*PI)

__host__ __device__ 
Vec getUnitVector(int d); // (y, x) // d in [0, 2*DIRECTIONS)

__host__ __device__
int getOrthogonalDirection(int d); // d in [0, 2*DIRECTIONS)

__host__ __device__ 
Vec getOrthogonalUnitVector(int d); // (y, x) // d in [0, 2*DIRECTIONS)

__host__ __device__
int getOppositeDirection(int d); // d in [0, 2*DIRECTIONS)

__host__ __device__
int getDirDifference(int d1, int d2); // d1, d2 in [0, DIRECTIONS)

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getDirColorRgb(int dir); // dir in [0, 2*DIRECTIONS)

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getDirColorLab(int dir); // dir in [0, 2*DIRECTIONS)

#endif