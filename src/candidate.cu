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
    float bestScore = -1.0f;
    float bestDir = 0.0f;
    for (int d = 0; d < DIRECTIONS; d++) {
        float score = computeLabScore(F, Fstep, y, x, d, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }
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



