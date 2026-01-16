#ifndef CAND_TYPE_HPP
#define CAND_TYPE_HPP

#include <opencv2/opencv.hpp>
#include <thrust/tuple.h>

#include "config.hpp"
#include "vec_type.hpp"
#include "directions.hpp"



struct Cand {
    double y, x;
    int dir;
    double score;

    Cand() = default;

    __host__ __device__
    Cand(double y, double x, int dir, double score);

    __host__ __device__
    bool operator<(const Cand& otherCand);

    __host__ __device__
    double distToLine(const Cand& otherCand);

    __host__ __device__
    static double dist(Cand& cand1, Cand& cand2);

    __host__ __device__
    static int dirDiff(Cand& cand1, Cand& cand2);

};

#endif