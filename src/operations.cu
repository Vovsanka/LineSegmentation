#include "operations.hpp"

cv::cuda::GpuMat uploadToGPU(const cv::Mat& cpuF) {
    cv::cuda::GpuMat gpuF;
    gpuF.upload(cpuF);
    return gpuF;
}

cv::Mat downloadToCPU(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    return cpuF;
}

void showImage(std::string name, const cv::Mat& cpuF) {
    cv::imshow(name, cpuF);
    cv::waitKey(0);
}

void showImage(std::string name, const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    showImage(name, cpuF);
}

void showMatrix(std::string name, const cv::Mat& cpuF) {
    cv::Mat Norm;
    cv::normalize(cpuF, Norm, 0, 255, cv::NORM_MINMAX);
    Norm.convertTo(Norm, CV_8U);
    showImage(name, Norm);
}

void showMatrix(std::string name, const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    showMatrix(name, cpuF);
}



