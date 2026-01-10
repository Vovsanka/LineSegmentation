#include "candidate.hpp"


__global__ 
void bestPixelScoreKernel(
    const uchar* F, size_t Fstep,
    double* S, size_t Sstep,
    int* D, size_t Dstep,
    int width, int height
) {
    int y = blockIdx.y;  
    int x = blockIdx.x;
    if (y >= height || x >= width) return;
    //
    int dir = threadIdx.x;
    if (dir >= DIRECTIONS) return;
    //
    int lane = dir & 31;
    int warpId = dir >> 5;
    //
    int warpCount = (DIRECTIONS + 31) / 32;
    //
    double score = computeLabScore(F, Fstep, y, x, dir, width, height);
    int bestDir = dir;
    // warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        double otherScore = __shfl_down_sync(0xffffffff, score, offset);
        int otherDir = __shfl_down_sync(0xffffffff, bestDir, offset);
        if (otherScore > score) {
            score = otherScore; 
            bestDir = otherDir;
        }
    }
    // shared memory for warp winners 
    extern __shared__ unsigned char smem[]; 
    double* warpMaxScore = (double*)smem; 
    int* warpMaxDir = (int*)(warpMaxScore + warpCount);
    if (lane == 0) {
        warpMaxScore[warpId] = score;
        warpMaxDir[warpId] = dir;
    } 
    __syncthreads();
    // final reduction by warp 0
    if (warpId == 0) {
        double finalScore = (lane < warpCount) ? warpMaxScore[lane] : -1e300;
        int finalDir = (lane < warpCount) ? warpMaxDir[lane] : -1;
        for (int offset = 16; offset > 0; offset >>= 1) {
            double otherScore = __shfl_down_sync(0xffffffff, finalScore, offset);
            int otherDir = __shfl_down_sync(0xffffffff, finalDir, offset);
            if (otherScore > finalScore) {
                finalScore = otherScore;
                finalDir = otherDir;
            } 
        } 
        if (lane == 0) {
            cell<double>(S, Sstep, y, x) = finalScore;
            cell<int>(D, Dstep, y, x) = finalDir;
        }
    }
}

__global__ 
void candidateThresholdKernel(
    const double *S, size_t Sstep,
    const int *D, size_t Dstep,
    uchar *C, size_t Cstep,
    int width, int height
) {                             
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    //
    double score = cell<double>(S, Sstep, y, x);
    //
    cell<uchar>(C, Cstep, y, x) = (score >= CAND_THRESHOLD) ? 1 : 0;
}



