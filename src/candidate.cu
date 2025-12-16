#include "candidate.hpp"


__global__ 
void candidatePreComputation(
    const uchar* F, size_t Fstep,
    float* S, size_t Sstep,
    int* D, size_t Dstep,
    int width, int height
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    // Compute the score matrix for every direction
    thrust::tuple<float,int> newScoreDir = bestPossibleScore(F, Fstep, y, x, width, height);
    float bestScore = thrust::get<0>(newScoreDir);
    int bestDir = thrust::get<1>(newScoreDir);
    //
    float* rowS = (float*)((uchar*)S + y * Sstep);
    rowS[x] = bestScore;
    //
    int* rowD = (int*)((uchar*)D + y * Dstep);
    rowD[x] = bestDir;
}

__global__ 
void candidateThresholdKernel(
    const float *S, size_t Sstep,
    const int *D, size_t Dstep,
    uchar *C, size_t Cstep,
    int width, int height
) {                             
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    //
    float* rowS = (float*)((uchar*)S + y * Sstep);
    float score = rowS[x];
    //
    uchar* rowC = (uchar*)((uchar*)C + y * Cstep);
    rowC[x] = (score >= CAND_THRESHOLD) ? 1 : 0;
}



