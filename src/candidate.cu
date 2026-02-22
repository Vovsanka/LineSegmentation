#include "candidate.hpp"


__global__
void bestPixelScoreKernelPixel(
    const uchar* F, size_t Fstep,
    double* S, size_t Sstep,
    int* D, size_t Dstep,
    int width, int height,
    bool beamScore
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    //
    Cand bestScoreDir = bestPossibleScoreDirection(F, Fstep, y, x, width, height, beamScore);
    double bestScore = bestScoreDir.score;
    int bestDir = bestScoreDir.dir;
    //
    cell<double>(S, Sstep, y, x) = bestScore;
    cell<int>(D, Dstep, y, x) = bestDir;
}

__host__
void computeBestPixelScores(
    cv::cuda::GpuMat& F,
    cv::cuda::GpuMat& S,
    cv::cuda::GpuMat& D,
    bool beamScore
) {
    int width = F.cols;
    int height = F.rows;
    //
    dim3 block(16, 16); // one thread for every pixel;
    //
    int gridX = (width + block.x - 1) / block.x;
    int gridY = (height + block.y - 1) / block.y;
    dim3 grid(gridX, gridY); 
    // 
    bestPixelScoreKernelPixel<<<grid, block>>>(
        F.ptr<uchar>(), F.step,
        S.ptr<double>(), S.step,
        D.ptr<int>(), D.step,
        F.cols, F.rows,
        beamScore
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



