#include <iostream>

#include "resize.hpp"
#include "candidate.hpp"



int main() {
    
    // Check the CUDA devices
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;

    // Load an RGB image
    cv::Mat originalF = cv::imread("../images/table.png", cv::IMREAD_COLOR);

    // Convert the image to LAB 
    cv::Mat labF;
    cv::cvtColor(originalF, labF, cv::COLOR_BGR2Lab);

    // Upload the image to GPU
    cv::cuda::GpuMat F;
    F.upload(labF);

    // Resize the image (interpolate for every half-pixel)
    cv::cuda::GpuMat zF = resize(F);

    // GPU threads for each pixel
    dim3 block(16, 16); // 256
    dim3 grid((zF.cols + block.x - 1) / block.x, (zF.rows + block.y - 1) / block.y); // round up to cover the whole image
    
    // compute the best scores for every pixel
    cv::cuda::GpuMat S(zF.size(), CV_64F);
    cv::cuda::GpuMat D(zF.size(), CV_32S);
    bestScoreKernel<<<grid, block>>>(
        zF.ptr<uchar>(), S.ptr<double>(), D.ptr<int>(),
        zF.cols, zF.rows
    );

    // choose the candidates
    cv::cuda::GpuMat C(zF.size(), CV_8U);
    candidateThresholdKernel<<<grid, block>>>(
        S.ptr<double>(), D.ptr<uchar>(), C.ptr<uchar>(),
        zF.cols, zF.rows
    );

    // show the image and the normalized matrices
    showImage(originalF);
    showImage(labF);
    showImage(zF);
    showMatrix(S);
    showMatrix(C);

    // // choose the candiates upgraded
    // cv::Mat cpuCI, cpuDI;
    // std::tie(cpuCI, cpuDI) = candidateIterativeSearch(cpuF, cpuS, cpuD);

    // showMatrix(cpuCI);

    return 0;
}