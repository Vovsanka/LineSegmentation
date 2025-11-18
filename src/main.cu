#include <iostream>

#include "best.hpp"



int main() {
    
    // Check the CUDA devices
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;

    // Load an RGB image
    cv::Mat cpuF = cv::imread("../images/table.png", cv::IMREAD_COLOR);

    // Score demo (also for subpixels)
    for (double y = 100.0; y <= 101.0; y += 0.1) {
        thrust::tuple<uchar,uchar,uchar> rgb = getRgbColors(cpuF.ptr(), y, 100, cpuF.cols, cpuF.rows);
        int r = thrust::get<0>(rgb);
        int g = thrust::get<1>(rgb);
        int b = thrust::get<2>(rgb);
        std::cout << y << ": " << r << " " << g << " " << b << " " << computeScore(cpuF.ptr(), y, 100.0, 0, cpuF.cols, cpuF.rows) << std::endl;
    }

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

    // // download the matrices to CPU
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

    return 0;
}