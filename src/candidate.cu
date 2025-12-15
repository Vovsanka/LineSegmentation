#include "candidate.hpp"


__global__ 
void bestScoreKernel(
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

// __global__ 
// void candidateThresholdKernel(const float *S, const int *D, uchar *C,
//                                          int width, int height) {
                                    
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= width || y >= height) return;
    
//     int idx = y * width + x;
//     float score = S[idx];
//     C[idx] = (score >= THRESHOLD) ? 1 : 0;
// }



