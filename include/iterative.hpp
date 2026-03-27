#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP


#include "cand_type.hpp"
#include "config.hpp"
#include "score.hpp"
#include "operations.hpp"


__host__ __device__
Cand upgradeCandidate(
    const uchar* F, size_t Fstep,
    cv::cuda::GpuMat& gpuF,
    Cand cand,
    int width, int height,
    bool beamScore = true
);

__host__
bool isBlocked(const uchar*B, size_t Bstep, double y, double x, int width, int height);

__host__
void setBlocked(uchar*B, size_t Bstep, double y, double x, int width, int height);


__host__
std::vector<Cand> candidateIterativeSearch(
    const uchar* F, size_t Fstep,
    cv::cuda::GpuMat& gpuF,
    std::vector<Cand>& tCandidates,
    int width, int height,
    bool beamScore = true
);

__host__ 
void candidateExpand(
    const uchar *F, size_t Fstep, 
    cv::cuda::GpuMat& gpuF,
    uchar* B, size_t Bstep,
    std::vector<Cand> &chosenCand,
    Cand cand,
    int invEdgeDir,
    double prevScore,
    int width, int height,
    bool beamScore = true
);

__global__
void bestPixelScoreKernelDirection(
    const uchar* F, size_t Fstep,
    double y, double x,
    double* scores,
    int width, int height,
    bool beamScore = true
);

__host__
Cand computeBestPixelCandidate(
    cv::cuda::GpuMat& F,
    double y, double x,
    bool beamScore = true
);


#endif