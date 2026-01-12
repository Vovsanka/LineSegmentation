#ifndef TYPES_HPP
#define TYPES_HPP

#include <opencv2/opencv.hpp>
#include <thrust/tuple.h>


struct Vec {
    double y, x;    

    Vec() = default;

    __host__ __device__
    Vec(double y, double x);
};


struct Cand {
    double y, x;
    int dir;
    double score;

    Cand() = default;

    __host__ __device__
    Cand(double y, double x, int dir, double score);

    __host__ __device__
    bool operator<(const Cand& otherCand);
};

#endif