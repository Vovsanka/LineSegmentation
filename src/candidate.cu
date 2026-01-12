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
    thrust::tuple<double,int> bestScoreDir = bestPossibleScore(F, Fstep, y, x, width, height);
    double bestScore = thrust::get<0>(bestScoreDir);
    int bestDir = thrust::get<1>(bestScoreDir);
    //
    cell<double>(S, Sstep, y, x) = bestScore;
    cell<int>(D, Dstep, y, x) = bestDir;
}

__host__
void computeBestPixelScores(
    cv::cuda::GpuMat& F,
    cv::cuda::GpuMat& S,
    cv::cuda::GpuMat& D
) {
    int width = S.cols;
    int height = S.rows;
    //
    dim3 block(16, 16); // one thread for every pixel;
    //
    int gridX = (width + block.x - 1) / block.x;
    int gridY = (height + block.y - 1) / block.y;
    dim3 grid(gridX, gridY); 
    // 
    bestPixelScoreKernel<<<grid, block>>>(
        F.ptr<uchar>(), F.step,
        S.ptr<double>(), S.step,
        D.ptr<int>(), D.step,
        F.cols, F.rows
    );
}

__host__ 
std::vector<Cand> extractSortedThresholdCandidates(
    cv::Mat& S, 
    cv::Mat& D
) {
    int width = S.cols;
    int height = S.rows;
    std::vector<Cand> candList;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double score = S.at<double>(y, x);
            if (score >= CAND_THRESHOLD) {
                double dir = D.at<int>(y, x);
                candList.push_back(Cand(y, x, dir, score));
            }
        }
    }
    //
    std::sort(std::begin(candList), std::end(candList));
    return candList;
}



