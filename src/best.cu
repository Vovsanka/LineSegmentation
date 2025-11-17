#include "best.hpp"


__global__ 
void bestScoreKernel(const uchar* F, double* S, int* D,
                                int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    // Compute the score matrix for every direction
    double bestScore = -1;
    double bestDir = 0;

    for (int d = 0; d < DIRECTIONS; ++d) {
        auto [unitY, unitX] = directionNormalUnitVector(d);
        double score = computeScore(F, y, x, unitX, unitY, width, height);
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
    C[idx] = (score >= CAND_SCORE) ? 1 : 0;
}


__host__ 
void rgbRun() {
    // Check the CUDA devices
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;

    // Load an RGB image
    cv::Mat cpuF = cv::imread("../images/table.png", cv::IMREAD_COLOR);
    
    // Upload the image to GPU
    cv::cuda::GpuMat F;
    F.upload(cpuF);

    // GPU threads for each pixel
    dim3 block(16, 16); // 256
    dim3 grid((F.cols + block.x - 1) / block.x, (F.rows + block.y - 1) / block.y); // round up to cover the whole image
    
    // compute the best scores for every pixel
    cv::cuda::GpuMat S(F.size(), CV_64F);
    cv::cuda::GpuMat D(F.size(), CV_32S);
    bestScoreKernel<<<grid, block>>>(
        F.ptr<uchar>(), S.ptr<double>(), D.ptr<int>(),
        F.cols, F.rows
    );

    // download the matrices to CPU
    cv::Mat cpuS, cpuD;
    S.download(cpuS);
    D.download(cpuD);

    // choose the candidates
    cv::cuda::GpuMat C(F.size(), CV_8U);
    candidateThresholdKernel<<<grid, block>>>(
        S.ptr<double>(), D.ptr<uchar>(), C.ptr<uchar>(),
        F.cols, F.rows
    );

    // download the matrices to CPU
    cv::Mat cpuC;    
    C.download(cpuC);

    // show the images 
    showImage(cpuF);
    showMatrix(cpuS);
    showMatrix(cpuC);

    // // choose the candiates upgraded
    // cv::Mat cpuCI, cpuDI;
    // std::tie(cpuCI, cpuDI) = candidateIterativeSearch(cpuF, cpuS, cpuD);

    // showMatrix(cpuCI);
}