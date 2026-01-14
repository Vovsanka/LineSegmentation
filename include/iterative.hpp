#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "types.hpp"
#include "config.hpp"
#include "score.hpp"
#include "operations.hpp"


__host__ __device__
Cand upgradeCandidate(
    const uchar* F, size_t Fstep,
    Cand cand,
    int width, int height
);

__host__
bool isBlocked(const uchar*B, size_t Bstep, double y, double x, int width, int height);

__host__
void setBlocked(uchar*B, size_t Bstep, double y, double x, int width, int height);


__host__
std::vector<Cand> candidateIterativeSearch(
    const uchar* F, size_t Fstep,
    const std::vector<Cand>& tCandidates,
    int width, int height
);

__host__ 
void candidateExpand(
    const uchar *F, size_t Fstep, 
    uchar* B, size_t Bstep,
    std::vector<Cand> &chosenCand,
    Cand cand,
    int invEdgeDir, double prevScore,
    int width, int height
);

__global__
void bestPixelScoreKernelDirection(
    const uchar* F, size_t Fstep,
    double y, double x,
    double* scores,
    int width, int height
);

__host__
Cand computeBestPixelScore(
    cv::cuda::GpuMat& F,
    double y, double x
);


#endif