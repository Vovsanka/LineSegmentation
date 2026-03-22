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

cv::Mat convertBGRtoLab(const cv::Mat& cpuF) {
    cv::Mat labF;
    cv::cvtColor(cpuF, labF, cv::COLOR_BGR2Lab);
    return labF;
}

cv::Mat convertBGRtoGrayscale(const cv::Mat& cpuF) {
    cv::Mat grayF;
    cv::cvtColor(cpuF, grayF, cv::COLOR_BGR2GRAY);
    return grayF;
}


// double computeScale(const cv::Mat& cpuF) {
//     return std::min(1.0*MAX_SIDE/cpuF.cols, 1.0*MAX_SIDE/cpuF.rows);
// }

// cv::Mat resizeDown(const cv::Mat& cpuF, double scale) {
//     cv::Size size(std::round(scale*cpuF.cols), std::round(scale*cpuF.rows));
//     cv::Mat scaledF;
//     cv::resize(cpuF, scaledF, size, 0, 0, cv::INTER_AREA); // clamp-to-edge strategy
//     return scaledF;
// }

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



