#include "candidate.hpp"


__global__
void bestPixelScoreKernel(
    const uchar* F, size_t Fstep,
    double* S, size_t Sstep,
    int* D, size_t Dstep,
    int width, int height
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    //
    int bestDir = 0;
    double bestScore = computeLabScore(F, Fstep, y, x, bestDir, width, height);
    //
    for (int dir = 1; dir < DIRECTIONS; dir++) {
        double score = computeLabScore(F, Fstep, y, x, dir, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = dir;
        }
    }
    //
    cell<double>(S, Sstep, y, x) = bestScore;
    cell<int>(D, Dstep, y, x) = bestDir;
}


__global__ 
void candidateThresholdKernel(
    const double *S, size_t Sstep,
    const int *D, size_t Dstep,
    uchar *C, size_t Cstep,
    int width, int height
) {                             
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    //
    double score = cell<double>(S, Sstep, y, x);
    //
    cell<uchar>(C, Cstep, y, x) = (score >= CAND_THRESHOLD) ? 1 : 0;
}



