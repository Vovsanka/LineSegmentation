#include "show.hpp"

__host__
cv::Mat downloadToCpu(const cv::cuda::GpuMat& gpuF) {
    cv::Mat F;
    gpuF.download(F);
    return F;
}

__host__
void showMatrix(const cv::cuda::GpuMat& gpuF) {
    cv::Mat F = downloadToCpu(gpuF);
    cv::Mat fNorm;
    cv::normalize(F, fNorm, 0, 255, cv::NORM_MINMAX);
    fNorm.convertTo(fNorm, CV_8U);
    cv::imshow("Matrix", fNorm);
    cv::waitKey(0);
}

__host__
void showMatrix(const cv::Mat &F) {
    cv::Mat fNorm;
    cv::normalize(F, fNorm, 0, 255, cv::NORM_MINMAX);
    fNorm.convertTo(fNorm, CV_8U);
    cv::imshow("Matrix", fNorm);
    cv::waitKey(0);
}

__host__
void showImage(const cv::cuda::GpuMat& gpuF) {
    cv::Mat F = downloadToCpu(gpuF);
    cv::imshow("Image", F);
    cv::waitKey(0);
}

__host__
void showImage(const cv::Mat& F) {
    cv::imshow("Image", F);
    cv::waitKey(0);
}

