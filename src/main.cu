#include <iostream>

#include <opencv2/opencv.hpp>

#include "operations.hpp"
#include "score.hpp"
#include "candidate.hpp"
#include "iterative.hpp"



int main() {

    // Check the CUDA devices (GPU support)
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;
    if (!cudaCount) return 1;

    // Load an RGB image
    // cv::Mat originalF = cv::imread("../images/black.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/mini-table.png", cv::IMREAD_COLOR);
    cv::Mat originalF = cv::imread("../images/table.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb1.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb2.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb3.png", cv::IMREAD_COLOR);
    if (originalF.empty()) return 1;
    std::cout << "Original image size: " << originalF.cols << "x" << originalF.rows << std::endl; 
    showImage(originalF);

    // Convert the image to the LAB color space
    cv::Mat labF = convertBGRtoLab(originalF);
    // showImage(labF);

    // apply the noise filtering to LAB-image (preserves the edge perception)
    cv::Mat filteredF = filterNoise(labF);
    // showImage(filteredF);

    // resize the image (to the reasonable processing size)
    double scale = computeScale(filteredF);
    cv::Mat scaledF = resize(filteredF, scale);
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << "Scaled image size: " << scaledF.cols << "x" << scaledF.rows << std::endl; 
    // showImage(scaledF);
    
    // Upload the image to GPU
    cv::cuda::GpuMat F = uploadToGPU(scaledF);
    // showImage(F);

    // GPU threads for each pixel
    dim3 block(16, 16); // 256
    dim3 grid((F.cols + block.x - 1)/block.x, (F.rows + block.y - 1)/block.y); // round up to cover the whole image
    
    // compute the best scores for every pixel
    cv::cuda::GpuMat S(F.size(), CV_64F);
    cv::cuda::GpuMat D(F.size(), CV_32S);
    candidatePreComputation<<<grid, block>>>(
        F.ptr<uchar>(), F.step,
        S.ptr<double>(), S.step,
        D.ptr<int>(), D.step,
        F.cols, F.rows
    );
    showMatrix(S);

    // choose the candidates
    cv::cuda::GpuMat C(F.size(), CV_8U);
    candidateThresholdKernel<<<grid, block>>>(
        S.ptr<double>(), S.step,
        D.ptr<int>(), D.step,
        C.ptr<uchar>(), C.step,
        F.cols, F.rows
    );
    showMatrix(C);

    // iterative search for candidates
    cv::Mat Fcpu = downloadToCPU(F);
    cv::Mat Scpu = downloadToCPU(S);
    cv::Mat Dcpu = downloadToCPU(D);
    
    std::vector<std::tuple<double,double>> candidates = candidateIterativeSearch(
        Fcpu.ptr<uchar>(), Fcpu.step,
        Scpu.ptr<double>(), Scpu.step,
        Dcpu.ptr<int>(), Dcpu.step,
        F.cols, F.rows
    );

    return 0;
}