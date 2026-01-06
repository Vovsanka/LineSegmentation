#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"
#include "operations.hpp"


__host__ __device__
thrust::tuple<double,double,int> upgradeCandidate(
    const uchar* F, size_t Fstep,
    thrust::tuple<double,double,int> cand,
    int width, int height
);

__host__
std::vector<std::tuple<double,int,int>> sortThresholdCandidates(
    const double *S, size_t Sstep, 
    int width, int height
);

__host__
bool isBlocked(const uchar*B, size_t Bstep, double y, double x, int width, int height);

__host__
void setBlocked(uchar*B, size_t Bstep, double y, double x, int width, int height);

__host__
std::vector<std::tuple<double,double>> candidateIterativeSearch(
    const uchar* F, size_t Fstep,
    const double *S, size_t Sstep,
    const int *D, size_t Dstep,
    int width, int height
);

__host__
void candidateExpand(
    const uchar *F, size_t Fstep, 
    uchar* B, size_t Bstep,
    std::vector<std::tuple<double,double>> &chosenCand,
    thrust::tuple<double,double,int> cand,
    int width, int height
);

#endif