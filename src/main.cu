#include <iostream>

#include <opencv2/opencv.hpp>

#include "operations.hpp"
#include "score.hpp"
#include "candidate.hpp"
// #include "iterative.hpp"



int main() {

    // Check the CUDA devices (GPU support)
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;
    if (!cudaCount) return 1;

    // Load an RGB image
    cv::Mat originalF = cv::imread("../images/black.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/mini-table.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/table.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb1.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb2.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb3.png", cv::IMREAD_COLOR);
    if (originalF.empty()) return 1;
    std::cout << "Original image size: " << originalF.cols << "x" << originalF.rows << std::endl; 
    showImage(originalF);

    // Convert the image to the LAB color space
    cv::Mat labF = convertBGRtoLab(originalF);
    showImage(labF);

    // apply the noise filtering to LAB-image (preserves the edge perception)
    cv::Mat filteredF = filterNoise(labF);
    showImage(filteredF);
    // cv::Mat filteredF = labF;

    // resize the image (to the reasonable processing size)
    float scale = computeScale(filteredF);
    cv::Mat scaledF = resize(filteredF, scale);
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << "Scaled image size: " << scaledF.cols << "x" << scaledF.rows << std::endl; 
    showImage(scaledF);

    ///// debug start
    for(int d = 0; d < DIRECTIONS; d++) {
        std::cout << computeLabScore(scaledF.ptr<uchar>(), 160, 160, d, scaledF.cols, scaledF.rows) << std::endl;
    }
    std::cout << std::endl;
    std::cout << computeLabScore(scaledF.ptr<uchar>(), 100, 100, 3, scaledF.cols, scaledF.rows) << std::endl;
    std::cout << computeLabScore(scaledF.ptr<uchar>(), 100, 100, 0, scaledF.cols, scaledF.rows) << std::endl;
    std::cout << computeLabScore(scaledF.ptr<uchar>(), 200, 300, 3, scaledF.cols, scaledF.rows) << std::endl;
    std::cout << computeLabScore(scaledF.ptr<uchar>(), 200, 300, 0, scaledF.cols, scaledF.rows) << std::endl;
    ///// debug end
    
    // Upload the image to GPU
    cv::cuda::GpuMat F = uploadToGPU(scaledF);
    showImage(F);

    // GPU threads for each pixel
    dim3 block(16, 16); // 256
    dim3 grid((F.cols + block.x - 1) / block.x, (F.rows + block.y - 1) / block.y); // round up to cover the whole image
    
    // compute the best scores for every pixel
    cv::cuda::GpuMat S(F.size(), CV_64F);
    cv::cuda::GpuMat D(F.size(), CV_32S);
    bestScoreKernel<<<grid, block>>>(
        F.ptr<uchar>(), S.ptr<float>(), D.ptr<int>(),
        F.cols, F.rows
    );
    showMatrix(S);

    // // choose the candidates
    // cv::cuda::GpuMat C(F.size(), CV_8U);
    // candidateThresholdKernel<<<grid, block>>>(
    //     S.ptr<float>(), D.ptr<int>(), C.ptr<uchar>(),
    //     F.cols, F.rows
    // );
    // showMatrix(C);

    // // /////////
    // // cv::Mat CI = candidateIterativeSearch(
    // //     downloadToCpu(F).ptr<uchar>(),
    // //     downloadToCpu(S).ptr<float>(),
    // //     downloadToCpu(D).ptr<int>(),
    // //     F.cols, F.rows
    // // );
    // // showMatrix(CI);

    return 0;
}