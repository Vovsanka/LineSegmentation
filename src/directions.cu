#include "directions.hpp"

__host__ __device__
double getRad(int d) { // returns [0, 2*PI)
    d %= 2*DIRECTIONS;
    return d*PI/DIRECTIONS;
}

__host__ __device__ 
Vec getUnitVector(int d) { // (y, x) // d in [0, 2*DIRECTIONS)
    float rad = static_cast<float>(getRad(d));
    return Vec(sinf(rad), cosf(rad));
}

__host__ __device__
int getOrthogonalDirection(int d) { // d in [0, 2*DIRECTIONS)
    return (d + DIRECTIONS/2) % (2*DIRECTIONS);
}

__host__ __device__
int getOppositeDirection(int d) { // d in [0, 2*DIRECTIONS)
    return (d + DIRECTIONS) % (2*DIRECTIONS);
}

__host__ __device__ 
Vec getOrthogonalUnitVector(int d) { // (y, x) // d in [0, 2*DIRECTIONS)
    return getUnitVector(getOrthogonalDirection(d));
}