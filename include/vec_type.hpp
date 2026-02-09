#ifndef VEC_TYPE_HPP
#define VEC_TYPE_HPP

#include <opencv2/opencv.hpp>
#include <thrust/tuple.h>


struct Vec {
    double y, x;    

    Vec() = default;

    __host__ __device__
    Vec(double y, double x);

    __host__ __device__   
    double len();

    __host__ __device__    
    Vec subtract(const Vec& otherVec);

    __host__ __device__
    double dot(const Vec& otherVec);

    __host__ __device__
    Vec operator*(double k);
};

#endif