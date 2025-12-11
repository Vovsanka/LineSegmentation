#include <iostream>

#include <opencv2/opencv.hpp>

#include "operations.hpp"
// #include "candidate.hpp"
// #include "iterative.hpp"



int main() {

    // Check the CUDA devices (GPU support)
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;
    if (!cudaCount) return 1;

    // Load an RGB image
    cv::Mat originalF = cv::imread("../images/mini-table.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/table.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb1.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb2.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb3.png", cv::IMREAD_COLOR);
    showImage(originalF);

    // Convert the image to the LAB color space
    cv::Mat labF = convertBGRtoLab(originalF);
    showImage(labF);

    // apply the noise filtering to LAB-image (preserves the edge perception)
    cv::Mat filteredF = filterNoise(labF);
    
    // Upload the image to GPU
    cv::cuda::GpuMat F = uploadToGPU(labF);
    // showImage(F);

    // Filter the noise (keeping the edges)
    

    // // // Resize the image (interpolate for every half-pixel)
    // // cv::cuda::GpuMat zF = resize(F);
    // // cv::cuda::GpuMat zF = F;

    // ////////////
    // for (int y = 15; y < 20; y++) {
    //     for (int x = 15; x < 20; x++) {
    //         thrust::tuple<uchar,uchar,uchar> lab = getColorChannels(labF.ptr(), y, x, F.cols, F.rows);
    //         int l = thrust::get<0>(lab);
    //         int a = thrust::get<1>(lab);
    //         int b = thrust::get<2>(lab);
    //         std::cout << l << " " << a << " " << b << std::endl;
    //     }
    // }
    // std::cout << computeLabScore(labF.ptr(), 17, 17, 0, F.cols, F.rows) << std::endl;

    // // GPU threads for each pixel
    // dim3 block(16, 16); // 256
    // dim3 grid((F.cols + block.x - 1) / block.x, (F.rows + block.y - 1) / block.y); // round up to cover the whole image
    
    // // compute the best scores for every pixel
    // cv::cuda::GpuMat S(F.size(), CV_64F);
    // cv::cuda::GpuMat D(F.size(), CV_32S);
    // bestScoreKernel<<<grid, block>>>(
    //     F.ptr<uchar>(), S.ptr<double>(), D.ptr<int>(),
    //     F.cols, F.rows
    // );
    // showMatrix(S);
    // // showMatrix(D);

    // // choose the candidates
    // cv::cuda::GpuMat C(F.size(), CV_8U);
    // candidateThresholdKernel<<<grid, block>>>(
    //     S.ptr<double>(), D.ptr<int>(), C.ptr<uchar>(),
    //     F.cols, F.rows
    // );
    // showMatrix(C);

    // // /////////
    // // cv::Mat CI = candidateIterativeSearch(
    // //     downloadToCpu(F).ptr<uchar>(),
    // //     downloadToCpu(S).ptr<double>(),
    // //     downloadToCpu(D).ptr<int>(),
    // //     F.cols, F.rows
    // // );
    // // showMatrix(CI);

    return 0;
}