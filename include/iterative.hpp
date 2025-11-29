#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "score.hpp"


__host__ __device__
thrust::tuple<double,double,double> upgradeCandidate(const uchar* F,
                                                     thrust::tuple<double,double,double>,
                                                     int width, int height);

__host__
std::vector<std::tuple<double,double,double>> sortThresholdCandidates(const double *S, int width, int height);


__host__
void candidateIterativeSearch(const uchar* F, const double *S, const double *D, int width, int height);



#endif