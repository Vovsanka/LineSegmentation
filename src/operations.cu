#include "operations.hpp"

__host__
dim3 getGrid(int width, int height) {
    int gridX = (width + GPU_BLOCK.x - 1) / GPU_BLOCK.x;
    int gridY = (height + GPU_BLOCK.y - 1) / GPU_BLOCK.y;
    return dim3(gridX, gridY); 
}

__host__
cv::cuda::GpuMat uploadToGPU(const cv::Mat& cpuF) {
    cv::cuda::GpuMat gpuF;
    gpuF.upload(cpuF);
    return gpuF;
}

__host__
cv::Mat downloadToCPU(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    return cpuF;
}

__host__
cv::Mat convertBGRtoLab(const cv::Mat& cpuF) {
    cv::Mat labF;
    cv::cvtColor(cpuF, labF, cv::COLOR_BGR2Lab);
    return labF;
}

__host__
cv::Mat filterNoise(const cv::Mat& cpuF) {
    cv::Mat filteredF;
    bilateralFilter(cpuF, filteredF, 5, 20, 20);
    return filteredF;
}

__host__
double computeScale(const cv::Mat& cpuF) {
    return std::min(1.0*MAX_SIDE/cpuF.cols, 1.0*MAX_SIDE/cpuF.rows);
}

__host__
cv::Mat resize(const cv::Mat& cpuF, double scale) {
    cv::Size size(std::round(scale*cpuF.cols), std::round(scale*cpuF.rows));
    cv::Mat scaledF;
    cv::resize(cpuF, scaledF, size, 0, 0, cv::INTER_CUBIC); // clamp-to-edge strategy
    return scaledF;
}

__host__
void showImage(const cv::Mat& cpuF) {
    cv::imshow("", cpuF);
    cv::waitKey(0);
}

__host__
void showImage(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    showImage(cpuF);
}

__host__
void showMatrix(const cv::Mat& cpuF) {
    cv::Mat Norm;
    cv::normalize(cpuF, Norm, 0, 255, cv::NORM_MINMAX);
    Norm.convertTo(Norm, CV_8U);
    showImage(Norm);
}

__host__
void showMatrix(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    showMatrix(cpuF);
}

__host__
void showScoreDirectionMatrix(
    cv::Mat& S,
    cv::Mat& D
) {
    int width = S.cols;
    int height = S.rows;
    cv::Mat Mlab(height, width, CV_8UC3); // LAB
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double score = S.at<double>(y, x);
            int dir = D.at<int>(y, x);
            Vec unitVector = getUnitVector(dir);
            int l = round(score*255.0);
            int a = round(unitVector.x*255.0);
            int b = round(unitVector.y*255.0);
            Mlab.at<cv::Vec3b>(y, x) = cv::Vec3b(l, a, b);
        }
    }
    cv::Mat Mbgr;
    cv::cvtColor(Mlab, Mbgr, cv::COLOR_Lab2BGR);
    showImage(Mbgr);
}






