#ifndef DIRECTIONS_HPP
#define DIRECTIONS_HPP

#include "types.hpp"
#include "config.hpp"


__host__ __device__
double getRad(int d); // returns [0, 2*PI)

__host__ __device__ 
Vec getUnitVector(int d); // (y, x) // d in [0, 2*DIRECTIONS)

__host__ __device__
int getOrthogonalDirection(int d); // d in [0, 2*DIRECTIONS)

__host__ __device__ 
Vec getOrthogonalUnitVector(int d); // (y, x) // d in [0, 2*DIRECTIONS)

__host__ __device__
int getOppositeDirection(int d); // d in [0, 2*DIRECTIONS)


#endif