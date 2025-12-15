#include "candidate.hpp"


__global__ 
void bestScoreKernel(const uchar* F, float* S, int* D,
                                int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    // Compute the score matrix for every direction
    float bestScore = -1;
    float bestDir = 0;

    for (int d = 0; d < DIRECTIONS; d++) {
        float score = computeLabScore(F, y, x, d, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }

    int idx = y * width + x;
    S[idx] = bestScore;
    D[idx] = bestDir;
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



