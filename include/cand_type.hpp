#ifndef CAND_TYPE_HPP
#define CAND_TYPE_HPP

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
    double distToLine(const Cand& otherCand) const;

    __host__ __device__
    static double dist(const Cand& cand1, const Cand& cand2);

    __host__ __device__
    static int dirDiff(const Cand& cand1, const Cand& cand2);

    __host__ __device__
    static bool positionComparator(const Cand& c1, const Cand& c2);
};

#endif