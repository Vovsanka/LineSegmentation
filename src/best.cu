#include "best.hpp"


// __device__
// void testDevice() {
//     double t0 = 255;
//     int t1 = 200;
//     t0 = fmax(t0, static_cast<double>(t1));
// }

__global__ 
void bestScoreKernel(const uchar* F, double* S, int* D,
                                int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // testDevice();
    
    // Compute the score matrix for every direction
    double bestScore = -1;
    double bestDir = 0;

    for (int d = 0; d < DIRECTIONS; d++) {
        double score = computeScore(F, y, x, d, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }

    int idx = y * width + x;
    S[idx] = bestScore;
    D[idx] = bestDir;
}

__global__ 
void candidateThresholdKernel(const double *S, const uchar *D, uchar *C,
                                         int width, int height) {
                                    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    double score = S[idx];
    C[idx] = (score >= THRESHOLD) ? 1 : 0;
}



